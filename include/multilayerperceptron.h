#include <Eigen/Dense>
#include <random>
#include <tuple>
#include "filehandler.h"
#include "activation_functions.h"
template <typename ActivationFunc>
class EigenLayer {
private:
	ActivationFunc func_;
public:
	using VecT = Eigen::VectorXd;
	using MatrT = Eigen::MatrixXd;

	VecT activations_;
	VecT gradients_;
	VecT biases_;
	MatrT weights_;

	EigenLayer(std::mt19937& rng, const std::size_t NNeurons, const std::size_t NBeforeNeurons) : activations_{VecT::Zero(NNeurons)}, gradients_{ VecT::Zero(NNeurons) }, biases_{ VecT::Zero(NNeurons) }, weights_{ MatrT::Zero(NNeurons, NBeforeNeurons) } {
		std::uniform_real_distribution dis(-1.0, 1.0);
		for (std::size_t i = 0; i < NNeurons; ++i) {
			activations_(i) = dis(rng);
		}

		for (std::size_t i = 0; i < NNeurons; ++i) {
			activations_(i) = dis(rng);
		}

		for (std::size_t i = 0; i < NNeurons; ++i) {
			for (std::size_t j = 0; j < NBeforeNeurons; ++j) {
				weights_(i, j) = dis(rng);
			}
		}
	}

	void forward_propagate(Eigen::Ref<const VecT> prev_activations) {
		activations_ = (weights_ * prev_activations) + biases_;
		for (int i = 0; i < activations_.size(); ++i) {
			gradients_(i) = func_.gradient(activations_(i));
			activations_(i) = func_.function(activations_(i));
		}
	}
};

namespace {
	size_t argmax(Eigen::Ref<const Eigen::VectorXd> input) {
		// Calculate the index of the maximum element in this Eigen vector.

		auto current_max = std::numeric_limits<double>::min();
		size_t max_idx = 0;
		for (int i = 0; i < input.size(); ++i) {
			if (input(i) > current_max) {
				current_max = input(i);
				max_idx = i;
			}
		}
		return max_idx;
	}
}

template <typename ActivationFunc>
class MultiLayerPerceptron {
public:

	std::vector<EigenLayer<ActivationFunc>> layers_;

	MultiLayerPerceptron(std::mt19937& rng, const std::vector<std::size_t>& layer_sizes) {
		layers_.emplace_back(rng, layer_sizes[0], 0);
		for (size_t i = 1; i < std::size(layer_sizes); ++i) {
			layers_.emplace_back(rng, layer_sizes[i], layer_sizes[i-1]);
		}
	}

	void forward_propagate(Eigen::VectorXd input) {
		layers_[0].activations_ = input;
		for (size_t i = 1; i < std::size(layers_); ++i) {
			layers_[i].forward_propagate(layers_[i - 1].activations_);
		}
	}

	Eigen::VectorXd predict(Image& input) {
		forward_propagate(input.flatten().cast<double>());
		return layers_[layers_.size() - 1].activations_;
	}

	void backward_propagate(Image& input, Eigen::Ref<const Eigen::VectorXd> true_output, const double learning_rate = 0.01) {
		Eigen::VectorXd predicted = predict(input);
		auto last_idx = layers_.size() - 1;
		Eigen::VectorXd deltas = 2.0*(true_output - predicted).cwiseProduct(layers_[last_idx].gradients_);
		for (size_t layer_idx = layers_.size() - 2; layer_idx > 0; layer_idx--) {			
			Eigen::MatrixXd outward_weights = layers_[layer_idx + 1].weights_;
			Eigen::MatrixXd weight_changes = deltas * layers_[layer_idx].activations_.transpose();
			layers_[layer_idx + 1].weights_ += weight_changes * learning_rate;
			layers_[layer_idx + 1].biases_ += deltas * learning_rate;
			
			deltas = (deltas.transpose() * outward_weights);
			deltas = deltas.cwiseProduct(layers_[layer_idx].gradients_);
		}
	}

	std::pair<double, double> fit(std::vector < std::pair<Image, Eigen::VectorXd> >& batch, const double learning_rate = 0.01) {
		//! Train the network to best predict images from this batch.
		/**
		 *	@param batch a batch of image, one-hot label vectors to train on.
		 *  @param learning_rate the rate at which to learn these images.
		 *  @return cost, accuracy two metrics to track progress.
		 */
		auto cost = 0.0;
		auto accuracy = 0.0;
		for (auto& [img, label] : batch) {
			Eigen::VectorXd predicted = predict(img);
			cost = (predicted - label).squaredNorm();
			if (argmax(predicted) == argmax(label)) {
				accuracy += 1;
			}

			backward_propagate(img, label, learning_rate);
		}
		return { cost / batch.size(), accuracy / batch.size() };
	}

	std::tuple<double, double, double, double > fit(std::vector < std::pair<Image, Eigen::VectorXd> >& batch, std::vector < std::pair<Image, Eigen::VectorXd> >& val_batch, const double learning_rate = 0.01) {
		//! Train the network to best predict images from this batch
		//! This is an overload that also takes validation data.
		/**
		 *	@param batch a batch of image, one-hot label vectors to train on.
		 *  @param learning_rate the rate at which to learn these images.
		 *  @return cost, accuracy, val_cost, val_accuracy
		 */
		auto cost = 0.0;
		auto accuracy = 0.0;
		for (auto& [img, label] : batch) {
			Eigen::VectorXd predicted = predict(img);
			cost += (predicted - label).squaredNorm();
			if (argmax(predicted) == argmax(label)) {
				accuracy += 1;
			}

			backward_propagate(img, label, learning_rate);
		}

		auto val_cost = 0.0;
		auto val_accuracy = 0.0;
		for (auto& [img, label]:val_batch) {
			Eigen::VectorXd predicted = predict(img);
			val_cost += (predicted - label).squaredNorm();
			if (argmax(predicted) == argmax(label)) {
				val_accuracy += 1;
			}
		}
		return { cost / batch.size(), accuracy / batch.size(), val_cost / val_batch.size(), val_accuracy / val_batch.size() };
	}
};
