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
    anymal_->setName("go1");
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
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(20.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.5);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 239;
    criticObDim_ = 248;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_); c_obDouble_.setZero(criticObDim_);

    prevAction_.setZero(nJoints_); targState_.setZero(37); currentState_.setZero(37); vtargState_.setZero(37);

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
//    footIndices_.insert(anymal_->getBodyIdx("FL_thigh"));
//    footIndices_.insert(anymal_->getBodyIdx("FR_thigh"));
//    footIndices_.insert(anymal_->getBodyIdx("RL_thigh"));
//    footIndices_.insert(anymal_->getBodyIdx("RR_thigh"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(anymal_);
      anymal_vref_ = server_->addVisualArticulatedSystem("v_ref", resourceDir_+"/go1/go1.urdf", 0,0,0,0.5);
      // posSphere_ = server_->addVisualSphere("debugSphere", 0.03, 0,1,0,1);
    }
  }

  void init() final { }

  void reset() final {
    t = 0;
    gc_init_ << 0, 0, 0.35, 1.0, 0.0, 0.0, 0.0, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5;
    gv_init_.setZero(gvDim_);

    if (gait_num_ == -1) t = bf_init_t;

    gc_init_.segment(0,2) = reference_.row(t).segment(0,2);
    gc_init_.segment(3,1) = reference_.row(t).segment(6,1);
    gc_init_.segment(4,3) = reference_.row(t).segment(3,3);
    if (gait_num_ == -1){
        gc_init_(2) = reference_.row(t)(2);
        gc_init_.segment(7,12) = reference_.row(t).segment(7,12);
        gv_init_ = reference_.row(t).tail(18);
    }

    anymal_->setState(gc_init_, gv_init_);

    anymal_kinematics_->setGeneralizedCoordinate(gc_); raisim::Vec<3> test;
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("FL_foot_fixed"), test);
    FL_footpos = test.e();
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("FR_foot_fixed"), test);
    FR_footpos = test.e();
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("RL_foot_fixed"), test);
    RL_footpos = test.e();
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("RR_foot_fixed"), test);
    RR_footpos = test.e();

    updateObservation();
    currentState_ << gc_, gv_;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) {
          if (isRendering_) std::this_thread::sleep_for(std::chrono::duration<double>(simulation_dt_));
          server_->unlockVisualizationServerMutex();
      }
    }

    anymal_kinematics_->setGeneralizedCoordinate(gc_); raisim::Vec<3> test;
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("FL_foot_fixed"), test);
    FL_footpos = test.e();
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("FR_foot_fixed"), test);
    FR_footpos = test.e();
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("RL_foot_fixed"), test);
    RL_footpos = test.e();
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("RR_foot_fixed"), test);
    RR_footpos = test.e();

    raisim::Vec<4> quat, des_quat;
    raisim::Mat<3,3> rot, des_rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    des_quat[0] = targState_(3); des_quat[1] = targState_(4); des_quat[2] = targState_(5); des_quat[3] = targState_(6);
    raisim::quatToRotMat(quat, rot); raisim::quatToRotMat(des_quat, des_rot);

    footposDiff << 0, 0, 0, 0;
    Eigen::Vector3d base_, base_ref;
    base_ << gc_(0), gc_(1), gc_(2); base_ref << targState_(0), targState_(1), targState_(2);

    anymal_kinematics_->setGeneralizedCoordinate(targState_.head(19));
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("FL_foot_fixed"), test);
    footposDiff(0) = ( des_rot.e().transpose()*(test.e() - base_ref) - rot.e().transpose()*(FL_footpos - base_) ).norm();
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("FR_foot_fixed"), test);
    footposDiff(1) = ( des_rot.e().transpose()*(test.e() - base_ref) - rot.e().transpose()*(FR_footpos - base_) ).norm();
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("RL_foot_fixed"), test);
    footposDiff(2) = ( des_rot.e().transpose()*(test.e() - base_ref) - rot.e().transpose()*(RL_footpos - base_) ).norm();
    anymal_kinematics_->getFramePosition(anymal_kinematics_->getFrameIdxByName("RR_foot_fixed"), test);
    footposDiff(3) = ( des_rot.e().transpose()*(test.e() - base_ref) - rot.e().transpose()*(RR_footpos - base_) ).norm();

    posDiff = gc_.head(3) - targState_.segment(0,3);
    dofDiff = gc_.tail(12) - targState_.segment(7,12);
    linvelDiff = gv_.head(3) - targState_.segment(19,3);
    angvelDiff = gv_.segment(3,3) - targState_.segment(22,3);
    dofvelDiff = gv_.tail(12) - targState_.tail(12);

    Eigen::VectorXd jointScale(12), bf_dofDiff(12);
    jointScale << 0.7, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 1.5, 1.5, 1.0, 1.5, 1.5;
    bf_dofDiff = dofDiff.array() * jointScale.array();

    Eigen::Vector3d comScale, bf_posDiff;
    comScale << 0., 0., 5.; // 1., 1., 5.
    bf_posDiff = posDiff.array() * comScale.array();

    Eigen::Vector3d linvelScale, bf_linvelDiff;
    bf_linvelDiff = linvelDiff;
    linvelScale << 1., 0., 1.5;
    bf_linvelDiff = linvelDiff.array() * linvelScale.array(); // 합쳐서 할땐 튜닝 상황에 따라서 linvel scale 없애는거 고려

//    Eigen::Quaterniond q1 = Eigen::Quaterniond(gc_[3], gc_[4], gc_[5], gc_[6]);
//    Eigen::Quaterniond q2 = Eigen::Quaterniond(targState_[3], targState_[4], targState_[5], targState_[6]);
//    Eigen::Quaterniond quatDiff = q1 * q2.conjugate();
    Eigen::VectorXd quatDiff(4);
    quatDiff = gc_.segment(3,4) - targState_.segment(3,4);
    Eigen::VectorXd dofvel(12);
    dofvel = gv_.segment(6, 12);

    if (gait_num_ == -1){
//        rewards_.record("bf_compos", exp(-0.5 * bf_posDiff.norm()));
        rewards_.record("bf_compos", -log(1.0 * bf_posDiff.norm() + 1e-6));
        rewards_.record("bf_dofpos", exp(-1. * bf_dofDiff.norm()));
        rewards_.record("bf_footpos", exp(-0.5 * footposDiff.norm()));
        rewards_.record("bf_linvel", exp(-0.5 * bf_linvelDiff.norm()));
        rewards_.record("bf_angvel", -log(1.0 * angvelDiff.norm() + 1e-6));
//        rewards_.record("bf_angvel", exp(-0.5 * angvelDiff.norm()));
        rewards_.record("bf_quat", exp(-5 * quatDiff.norm()));
        rewards_.record("bf_dofvel", exp(-0.1 * dofvelDiff.norm()));

        rewards_.record("loco_torque", 0);
        rewards_.record("loco_compos", 0);
        rewards_.record("loco_footpos", 0);
        rewards_.record("loco_linvel", 0);
        rewards_.record("loco_angvel", 0);
        rewards_.record("loco_quat", 0);
    }
    else {
        rewards_.record("bf_compos", 0);
        rewards_.record("bf_dofpos", 0);
        rewards_.record("bf_footpos", 0);
        rewards_.record("bf_linvel", 0);
        rewards_.record("bf_angvel", 0);
        rewards_.record("bf_quat", 0);
        rewards_.record("bf_dofvel", 0);

        rewards_.record("loco_torque", anymal_->getGeneralizedForce().squaredNorm());
        rewards_.record("loco_compos", exp(-2 * posDiff.norm()));
        rewards_.record("loco_footpos", -log(0.5 * footposDiff.norm() + 1e-6));
        rewards_.record("loco_linvel", -log(1 * linvelDiff.norm() + 1e-6));
        rewards_.record("loco_angvel", -log(1 * angvelDiff.norm() + 1e-6));
        rewards_.record("loco_quat", exp(-0.5 * quatDiff.norm()));
        rewards_.record("loco_dofpos", exp(-1. * dofDiff.norm()));
    }


    prevAction_ = pTarget12_;
    t += 1;

    updateObservation();
    currentState_ << gc_, gv_;

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

    footContactBool_.setZero();
    for (auto &contact: anymal_->getContacts()) {
        if (contact.getlocalBodyIndex() == anymal_->getBodyIdx("FL_foot_fixed") ||
            contact.getlocalBodyIndex() == anymal_->getBodyIdx("FL_calf")) {
            footContactBool_(0) = 1;
        }
        if (contact.getlocalBodyIndex() == anymal_->getBodyIdx("FR_foot_fixed") ||
            contact.getlocalBodyIndex() == anymal_->getBodyIdx("FR_calf")) {
            footContactBool_(1) = 1;
        }
        if (contact.getlocalBodyIndex() == anymal_->getBodyIdx("RL_foot_fixed") ||
            contact.getlocalBodyIndex() == anymal_->getBodyIdx("RL_calf")) {
            footContactBool_(2) = 1;
        }
        if (contact.getlocalBodyIndex() == anymal_->getBodyIdx("RL_foot_fixed") ||
            contact.getlocalBodyIndex() == anymal_->getBodyIdx("RR_calf")) {
            footContactBool_(3) = 1;
        }
    }

//    thighContactBool_.setZero();
//    for (auto &contact: anymal_->getContacts()) {
//        if (contact.getlocalBodyIndex() == anymal_->getBodyIdx("FL_thigh")) {
//            thighContactBool_(0) = 1;
//        }
//        if (contact.getlocalBodyIndex() == anymal_->getBodyIdx("FR_thigh")) {
//            thighContactBool_(1) = 1;
//        }
//        if (contact.getlocalBodyIndex() == anymal_->getBodyIdx("RL_thigh")) {
//            thighContactBool_(2) = 1;
//        }
//        if (contact.getlocalBodyIndex() == anymal_->getBodyIdx("RR_thigh")) {
//            thighContactBool_(3) = 1;
//        }
//    }

    obDouble_ << gc_[2], /// body height
        gc_.segment(3,4), /// body orientation
        rot.e().row(2).transpose(), /// gravity vector
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12), /// joint velocity
        prevAction_, /// action history
        FL_footpos(2), FR_footpos(2), RL_footpos(2), RR_footpos(2), /// foot height

        reference_.row(t - 1 + 1).transpose(),
        reference_.row(t - 1 + 6).transpose(),
        reference_.row(t - 1 + 11).transpose(),
        reference_.row(t - 1 + 16).transpose(),
        reference_.row(t - 1 + 21).transpose();

    c_obDouble_ << gait_num_, /// skill num
        gc_.head(3), /// world frame body pos
        gc_.segment(3,4), /// base quaternion
        rot.e().row(2).transpose(), /// gravity vector
        gv_.segment(6,6), /// world frame lin & angvel
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12), /// joint velocity
        prevAction_, /// action history
        FL_footpos(2), FR_footpos(2), RL_footpos(2), RR_footpos(2), /// foot height
        reference_.row(t - 1 + 1).transpose(),
        reference_.row(t - 1 + 6).transpose(),
        reference_.row(t - 1 + 11).transpose(),
        reference_.row(t - 1 + 16).transpose(),
        reference_.row(t - 1 + 21).transpose();

    targState_ << reference_.row(t - 1 + 1).head(3).transpose(),
                  reference_.row(t - 1 + 1)(6),
                  reference_.row(t - 1 + 1).segment(3,3).transpose(),
                  reference_.row(t - 1 + 1).tail(30).transpose();

    vtargState_ << vreference_.row(t - 1 + 1).head(3).transpose(),
                vreference_.row(t - 1 + 1)(6),
                vreference_.row(t - 1 + 1).segment(3,3).transpose(),
                vreference_.row(t - 1 + 1).tail(30).transpose();

    if (server_) {
        if (gait_num_ == 1) anymal_vref_->setColor(0.8,0,0,0.5);
        else if (gait_num_ == 2) anymal_vref_->setColor(1.0, 1.0, 0, 0.5);
        else if (gait_num_ == 3) anymal_vref_->setColor(0,0.8,0,0.5);
        else if (gait_num_ == 0) anymal_vref_->setColor(0,0,0.8,0.5);
        else anymal_vref_->setColor(0.,0,0,0.5);
        /// in evaluation, set replace this with vtargState_
        anymal_vref_->setGeneralizedCoordinate(targState_);
    }
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  void observe_critic(Eigen::Ref<EigenVec> ob) {
      /// convert it to float
    ob = c_obDouble_.cast<float>();
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

  void flushTrajectory(const Eigen::MatrixXd& trajectory, const int gait_num = -1) {
    reference_ = trajectory;

    anymal_->getState(gc_,gv_);
    vreference_ = trajectory;
    /// use this line in evaluation
    //  vreference_.leftCols(2).rowwise() += gc_.head(2).transpose();

    gait_num_ = gait_num;
//    if (visualizable_) std::cout << "Gait number is " << gait_num_ << std::endl;
    t = 0;

    /// for backflip training
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis1(1, 5);
    std::uniform_int_distribution<> dis2(30, 35);
    std::uniform_real_distribution<> prob_dis(0.0, 1.0);
    double random_value = prob_dis(gen);

    if (random_value < 0.8){
        bf_init_t = 0;
    }
    else if (random_value < 0.9){
        bf_init_t = dis1(gen);
    }
    else {
        bf_init_t = dis2(gen);
    }

    updateObservation();
  }

  void isRendering(bool isRenderingNow) {
      isRendering_ = isRenderingNow;
  }

  void getState(Eigen::Ref<EigenVec> st) {
      st = currentState_.cast<float>();
  }

  int getModeNum(){
      return gait_num_;
  }

 private:
  int t = 0, bf_init_t = 0;
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* anymal_;
  raisim::ArticulatedSystem* anymal_kinematics_;
  raisim::ArticulatedSystemVisual* anymal_vref_;
  raisim::Visuals* posSphere_;

  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -9.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, c_obDouble_;
  Eigen::VectorXd prevAction_, targState_, currentState_, vtargState_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  Eigen::MatrixXd reference_; Eigen::MatrixXd vreference_;
  Eigen::Vector3d posDiff = Eigen::Vector3d::Zero();
  Eigen::VectorXd dofDiff = Eigen::VectorXd::Zero(12);
  Eigen::VectorXd dofvelDiff = Eigen::VectorXd::Zero(12);
  Eigen::Vector3d linvelDiff = Eigen::Vector3d::Zero();
  Eigen::Vector3d angvelDiff = Eigen::Vector3d::Zero();
  Eigen::Vector3d FL_footpos, FR_footpos, RL_footpos, RR_footpos;
  Eigen::VectorXd footposDiff = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd footContactBool_ = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd thighContactBool_ = Eigen::VectorXd::Zero(4);
  int gait_num_ = -1;
  bool isRendering_ = false;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

