//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "RaisimGymEnv.hpp"
#include "omp.h"
#include "Yaml.hpp"

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>

std::vector<Eigen::MatrixXd> globalMatrices;

// CSV 파일을 읽어 전역 벡터로 변환하는 함수
void readCSVtoEigen(const std::string& filename, const int episode_length = 250) {
  std::ifstream file(filename);
  std::string line;
//  globalMatrices.clear(); // 기존 데이터를 초기화
  const int rows_per_matrix = episode_length;
  const int cols_per_matrix = 37;

  if (file.is_open()) {
    int row_count = 0;
    Eigen::MatrixXd mat(rows_per_matrix, cols_per_matrix);
    Eigen::Vector2d first_row_vals(0,0);
    bool first_row_read = false;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        int col = 0;
        while (std::getline(lineStream, cell, ',')) {
            mat(row_count % rows_per_matrix, col) = std::stod(cell);
            col++;
        }

        // starting at (0,0)
        if (!first_row_read) {
            first_row_vals(0) = mat(row_count % rows_per_matrix, 0);
            first_row_vals(1) = mat(row_count % rows_per_matrix, 1);
            mat(row_count % rows_per_matrix, 0) -= first_row_vals(0);
            mat(row_count % rows_per_matrix, 1) -= first_row_vals(1);
            first_row_read = true;
        } else {
            mat(row_count % rows_per_matrix, 0) -= first_row_vals(0);
            mat(row_count % rows_per_matrix, 1) -= first_row_vals(1);
        }

        row_count++;
        if (row_count % rows_per_matrix == 0) {
            globalMatrices.push_back(mat);
            mat = Eigen::MatrixXd(rows_per_matrix, cols_per_matrix); // 새로운 행렬 초기화
            first_row_read = false;
        }
    }
    // 파일이 끝난 후, 남은 데이터가 있는 경우 처리
    if (row_count % rows_per_matrix != 0) {
      globalMatrices.push_back(mat.topRows(row_count % rows_per_matrix));
    }
    file.close();
  } else {
    std::cerr << "파일을 열 수 없습니다." << std::endl;
  }
}

namespace raisim {

int THREAD_COUNT;

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg, bool normalizeObservation=true)
      : resourceDir_(resourceDir), cfgString_(cfg), normalizeObservation_(normalizeObservation) {
    Yaml::Parse(cfg_, cfg);

    std::string CSVpath = resourceDir + "/data/trot_all.csv";
    readCSVtoEigen(CSVpath);
    trot_idx = globalMatrices.size();
    std::string CSVpath1 = resourceDir + "/data/bound_all.csv";
    readCSVtoEigen(CSVpath1);
    bound_idx = globalMatrices.size();
    std::string CSVpath2 = resourceDir + "/data/pace_all.csv";
    readCSVtoEigen(CSVpath2);
    pace_idx = globalMatrices.size();
    std::string CSVpath3 = resourceDir + "/data/pronk_all.csv";
    readCSVtoEigen(CSVpath3);
    pronk_idx = globalMatrices.size();

    std::string CSVpath4 = resourceDir + "/data/backflip__refined.csv";
    readCSVtoEigen(CSVpath4);

//    std::string CSVpath4 = resourceDir + "/data/backflip_.csv";
//    readCSVtoEigen(CSVpath4, 144);

    std::cout << "Total " << globalMatrices.size() << " Episodes" << std::endl;

    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
    init();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  const std::string& getResourceDir() const { return resourceDir_; }
  const std::string& getCfgString() const { return cfgString_; }

  void init() {
    THREAD_COUNT = cfg_["num_threads"].template As<int>();
    omp_set_num_threads(THREAD_COUNT);
    num_envs_ = cfg_["num_envs"].template As<int>();

    environments_.reserve(num_envs_);
    rewardInformation_.reserve(num_envs_);
    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
      rewardInformation_.push_back(environments_.back()->getRewards().getStdMap());
    }

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> distrib(0, globalMatrices.size() - 1);
      int random_index = distrib(gen);

      int gait_num;
      if (random_index < trot_idx) gait_num = 1;
      else if (random_index < bound_idx) gait_num = 2;
      else if (random_index < pace_idx) gait_num = 3;
      else if (random_index < pronk_idx) gait_num = 0;
      else gait_num = -1;

      environments_[i]->flushTrajectory(globalMatrices[random_index], gait_num);
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    criticObDim_ = environments_[0]->getCriticObDim();
    actionDim_ = environments_[0]->getActionDim();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")

    /// ob scaling
    if (normalizeObservation_) {
      obMean_.setZero(obDim_);
      obVar_.setOnes(obDim_);
      recentMean_.setZero(obDim_);
      recentVar_.setZero(obDim_);
      delta_.setZero(obDim_);
      epsilon.setZero(obDim_);
      epsilon.setConstant(1e-8);
    }

    if (normalizeObservation_) {
        cobMean_.setZero(criticObDim_);
        cobVar_.setOnes(criticObDim_);
        recentCMean_.setZero(criticObDim_);
        recentCVar_.setZero(criticObDim_);
        cdelta_.setZero(criticObDim_);
        cepsilon.setZero(criticObDim_);
        cepsilon.setConstant(1e-8);
    }

  }

  // resets all environments and returns observation
  void reset() {
    for (auto env: environments_) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> distrib(0, globalMatrices.size() - 1);
      int random_index = distrib(gen);

      int gait_num;
      if (random_index < trot_idx) gait_num = 1;
      else if (random_index < bound_idx) gait_num = 2;
      else if (random_index < pace_idx) gait_num = 3;
      else if (random_index < pronk_idx) gait_num = 0;
      else gait_num = -1;
      env->flushTrajectory(globalMatrices[random_index], gait_num);

      env->reset();
    }


  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));

    if (normalizeObservation_)
      updateObservationStatisticsAndNormalize(ob, updateStatistics);
  }

  void observe_critic(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe_critic(ob.row(i));

    if (normalizeObservation_)
      updateCriticObservationStatisticsAndNormalize(ob, updateStatistics);
  }


  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done);
  }

  void turnOnVisualization() {
    if(render_) {
      environments_[0]->turnOnVisualization();
      environments_[0]->isRendering(true);
    }
  }

  void turnOffVisualization() {
    if(render_) {
      environments_[0]->turnOffVisualization();
      environments_[0]->isRendering(false);
    }
  }

  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }
  void getObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float &count) {
    mean = obMean_; var = obVar_; count = obCount_; }
  void setObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    obMean_ = mean; obVar_ = var; obCount_ = count; }

  void getCriticObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float &count) {
      mean = cobMean_; var = cobVar_; count = cobCount_; }
  void setCriticObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
      cobMean_ = mean; cobVar_ = var; cobCount_ = count; }

  void getState(Eigen::Ref<EigenRowMajorMat> &st){
      for (int i = 0; i < num_envs_; i++)
          environments_[i]->getState(st.row(i));
  }

  void updateRef(Eigen::Ref<EigenRowMajorMat> &ref, int &gait_num){
      Eigen::MatrixXd convertedRef = ref.cast<double>();
      for (int i = 0; i < num_envs_; i++){
          environments_[i]->flushTrajectory(convertedRef, gait_num);
      }
  }

  void setSeed(int seed) {
    int seed_inc = seed;

    #pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++)
      environments_[i]->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return obDim_; }
  int getActionDim() { return actionDim_; }
  int getNumOfEnvs() { return num_envs_; }
  int getCriticObDim() { return criticObDim_; }

  int getVisEnvMode() {
      return environments_[0]->getModeNum();
  }

  ////// optional methods //////
  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

  const std::vector<std::map<std::string, float>>& getRewardInfo() { return rewardInformation_; }

 private:
  void updateObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
    if (updateStatistics) {
      recentMean_ = ob.colwise().mean();
      recentVar_ = (ob.rowwise() - recentMean_.transpose()).colwise().squaredNorm() / num_envs_;

      delta_ = obMean_ - recentMean_;
      for(int i=0; i<obDim_; i++)
        delta_[i] = delta_[i]*delta_[i];

      float totCount = obCount_ + num_envs_;

      obMean_ = obMean_ * (obCount_ / totCount) + recentMean_ * (num_envs_ / totCount);
      obVar_ = (obVar_ * obCount_ + recentVar_ * num_envs_ + delta_ * (obCount_ * num_envs_ / totCount)) / (totCount);
      obCount_ = totCount;
    }

#pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++)
      ob.row(i) = (ob.row(i) - obMean_.transpose()).template cwiseQuotient<>((obVar_ + epsilon).cwiseSqrt().transpose());
  }

  void updateCriticObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
      if (updateStatistics) {
          recentCMean_ = ob.colwise().mean();
          recentCVar_ = (ob.rowwise() - recentCMean_.transpose()).colwise().squaredNorm() / num_envs_;

          cdelta_ = cobMean_ - recentCMean_;
          for(int i=0; i<criticObDim_; i++)
              cdelta_[i] = cdelta_[i]*cdelta_[i];

          float totCount = cobCount_ + num_envs_;

          cobMean_ = cobMean_ * (cobCount_ / totCount) + recentCMean_ * (num_envs_ / totCount);
          cobVar_ = (cobVar_ * cobCount_ + recentCVar_ * num_envs_ + cdelta_ * (cobCount_ * num_envs_ / totCount)) / (totCount);
          cobCount_ = totCount;
      }

#pragma omp parallel for schedule(auto)
      for(int i=0; i<num_envs_; i++)
          ob.row(i) = (ob.row(i) - cobMean_.transpose()).template cwiseQuotient<>((cobVar_ + cepsilon).cwiseSqrt().transpose());
  }

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done) {
    reward[agentId] = environments_[agentId]->step(action.row(agentId));
    rewardInformation_[agentId] = environments_[agentId]->getRewards().getStdMap();

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }
  }

  std::vector<Eigen::MatrixXd> trajectories;
  int trot_idx = 0, bound_idx = 0, pace_idx = 0, pronk_idx = 0;
  int vis_mode;

  std::vector<ChildEnvironment *> environments_;
  std::vector<std::map<std::string, float>> rewardInformation_;

  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0;
  int criticObDim_ = 0;
  bool recordVideo_=false, render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;
  std::string cfgString_;

  /// observation running mean
  bool normalizeObservation_ = true;
  EigenVec obMean_;
  EigenVec obVar_;
  EigenVec cobMean_, cobVar_;
  EigenVec recentCMean_, recentCVar_, cdelta_, cepsilon;
  float obCount_ = 1e-4;
  float cobCount_ = 1e-4;
  EigenVec recentMean_, recentVar_, delta_;
  EigenVec epsilon;
};


class NormalDistribution {
 public:
  NormalDistribution() : normDist_(0.f, 1.f) {}

  float sample() { return normDist_(gen_); }
  void seed(int i) { gen_.seed(i); }

 private:
  std::normal_distribution<float> normDist_;
  static thread_local std::mt19937 gen_;
};
thread_local std::mt19937 raisim::NormalDistribution::gen_;


class NormalSampler {
 public:
  NormalSampler(int dim) {
    dim_ = dim;
    normal_.resize(THREAD_COUNT);
    seed(0);
  }

  void seed(int seed) {
    // this ensures that every thread gets a different seed
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < THREAD_COUNT; i++)
      normal_[0].seed(i + seed);
  }

  inline void sample(Eigen::Ref<EigenRowMajorMat> &mean,
                     Eigen::Ref<EigenVec> &std,
                     Eigen::Ref<EigenRowMajorMat> &samples,
                     Eigen::Ref<EigenVec> &log_prob) {
    int agentNumber = log_prob.rows();

#pragma omp parallel for schedule(auto)
    for (int agentId = 0; agentId < agentNumber; agentId++) {
      log_prob(agentId) = 0;
      for (int i = 0; i < dim_; i++) {
        const float noise = normal_[omp_get_thread_num()].sample();
        samples(agentId, i) = mean(agentId, i) + noise * std(i);
        log_prob(agentId) -= noise * noise * 0.5 + std::log(std(i));
      }
      log_prob(agentId) -= float(dim_) * 0.9189385332f;
    }
  }
  int dim_;
  std::vector<NormalDistribution> normal_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
