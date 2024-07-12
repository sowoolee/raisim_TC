//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <VectorizedEnvironment.hpp>
#include <Eigen/Dense>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/go1/go1.urdf");
    anymal_kinematics_= new ArticulatedSystem(resourceDir_+"/go1/go1.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.35, 1.0, 0.0, 0.0, 0.0, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 231;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    prevAction_.setZero(nJoints_); targState_.setZero(37);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("FL_foot_fixed"));
    footIndices_.insert(anymal_->getBodyIdx("FR_foot_fixed"));
    footIndices_.insert(anymal_->getBodyIdx("RL_foot_fixed"));
    footIndices_.insert(anymal_->getBodyIdx("RR_foot_fixed"));
    footIndices_.insert(anymal_->getBodyIdx("FL_calf"));
    footIndices_.insert(anymal_->getBodyIdx("FR_calf"));
    footIndices_.insert(anymal_->getBodyIdx("RL_calf"));
    footIndices_.insert(anymal_->getBodyIdx("RR_calf"));
    // footIndices_.insert(anymal_->getBodyIdx("FL_thigh"));
    // footIndices_.insert(anymal_->getBodyIdx("FR_thigh"));
    // footIndices_.insert(anymal_->getBodyIdx("RL_thigh"));
    // footIndices_.insert(anymal_->getBodyIdx("RR_thigh"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer(8088);
      server_->focusOn(anymal_);
    }
  }

  void init() final { }

  void reset() final {
    gc_init_.segment(0,2) = reference_.row(0).segment(0,2);
    gc_init_.segment(3,1) = reference_.row(0).segment(6,1);
    gc_init_.segment(4,3) = reference_.row(0).segment(3,3);
    anymal_->setState(gc_init_, gv_init_);
    anymal_kinematics_->setGeneralizedCoordinate(gc_init_);
    raisim::Vec<3> test;
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("FL_foot_fixed"),test);
    t = 0;
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    // Eigen::VectorXd targgc, targgv;
    // targgc.setZero(19); targgv.setZero(18);
    // targgc << reference_.row(t).head(2).transpose(),
    //           reference_.row(t)(2),
    //           reference_.row(t)(6),
    //           reference_.row(t).segment(3,3).transpose(),
    //           reference_.row(t).segment(7,12).transpose();
    // targgv = reference_.row(t).tail(18);
    // targgc = targState_.head(19);
    // targgv = targState_.tail(18);
    // anymal_->setState(targgc, targgv);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    posDiff = gc_.head(3) - targState_.segment(0,3);
    dofDiff = gc_.tail(12) - targState_.segment(7,12);
    linvelDiff = gv_.head(3) - targState_.segment(19,3);
    angvelDiff = gv_.segment(3,3) - targState_.segment(22,3);

    Eigen::Quaterniond q1 = Eigen::Quaterniond(gc_[3], gc_[4], gc_[5], gc_[6]);
    Eigen::Quaterniond q2 = Eigen::Quaterniond(targState_[3], targState_[4], targState_[5], targState_[6]);
    Eigen::Quaterniond quatDiff = q1 * q2.conjugate();

    rewards_.record("compos", exp(-2 * posDiff.norm()));
    rewards_.record("dofpos", exp(-2 * dofDiff.norm()));
    rewards_.record("linvel", exp(-2 * linvelDiff.norm()));
    rewards_.record("angvel", exp(-2 * angvelDiff.norm()));
    rewards_.record("quat", exp(-2 * quatDiff.norm()));

    prevAction_ = pTarget12_;
    t += 1;

    updateObservation();


    return rewards_.sum();
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12), /// joint velocity
        prevAction_,
        reference_.row(t - 1 + 1).transpose();
        reference_.row(t - 1 + 6).transpose(),
        reference_.row(t - 1 + 11).transpose(),
        reference_.row(t - 1 + 16).transpose(),
        reference_.row(t - 1 + 21).transpose();

    targState_ << reference_.row(t - 1 + 1).head(3).transpose(),
                  reference_.row(t - 1 + 1)(6),
                  reference_.row(t - 1 + 1).segment(3,3).transpose(),
                  reference_.row(t - 1 + 1).tail(30).transpose();

  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

  void flushTrajectory(const Eigen::MatrixXd& trajectory) {
    // 트래젝토리를 업데이트하는 로직을 여기에 추가합니다.
    reference_ = trajectory;
  }

 private:
  int t = 0;
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* anymal_;
  raisim::ArticulatedSystem* anymal_kinematics_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::VectorXd prevAction_, targState_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  Eigen::MatrixXd reference_;
  Eigen::Vector3d posDiff = Eigen::Vector3d::Zero();
  Eigen::VectorXd dofDiff = Eigen::VectorXd::Zero(12);
  Eigen::Vector3d linvelDiff = Eigen::Vector3d::Zero();
  Eigen::Vector3d angvelDiff = Eigen::Vector3d::Zero();

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

