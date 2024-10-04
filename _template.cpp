#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

float randomFloatInRange(float n, float m) {
    float random_number = static_cast<float>(rand()) / RAND_MAX;
    return n + random_number * (m - n);
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

class Neuron {
public:
    vector<float> weights;
    float bias;
    float output;
    float delta;

    Neuron(int number_of_inputs) {
        for (int i = 0; i < number_of_inputs; i++) {
            weights.emplace_back(randomFloatInRange(-1, 1));
        }
        bias = randomFloatInRange(-1, 1);
    }

    float forward(const vector<float> &inputs) {
        float weighted_sum = inner_product(inputs.begin(), inputs.end(), weights.begin(), 0.0f) + bias;
        output = sigmoid(weighted_sum);
        return output;
    }

    void updateWeights(const vector<float> &inputs, float learning_rate) {
        for (int i = 0; i < weights.size(); i++) {
            weights[i] += learning_rate * delta * inputs[i];
        }
        bias += learning_rate * delta;
    }
};

class Layer {
public:
    vector<Neuron> neurons;

    Layer(int number_of_neurons, int number_of_inputs_per_neuron) {
        for (int i = 0; i < number_of_neurons; i++) {
            Neuron neuron(number_of_inputs_per_neuron);
            neurons.emplace_back(neuron);
        }
    }

    vector<float> forward(const vector<float> &inputs) {
        vector<float> outputs;
        for (Neuron &neuron : neurons) {
            outputs.emplace_back(neuron.forward(inputs));
        }
        return outputs;
    }
};

class NeuralNetwork {
public:
    vector<Layer> layers;
    float learning_rate;

    NeuralNetwork(const vector<int> &layer_sizes, float lr) : learning_rate(lr) {
        for (int i = 1; i < layer_sizes.size(); i++) {
            layers.emplace_back(Layer(layer_sizes[i], layer_sizes[i - 1]));
        }
    }

    vector<float> forward(vector<float> inputs) {
        for (Layer &layer : layers) {
            inputs = layer.forward(inputs);
        }
        return inputs;
    }

    void backpropagate(const vector<float> &inputs, const vector<float> &expected_output) {
        // Calculate output layer delta
        Layer &output_layer = layers.back();
        for (int i = 0; i < output_layer.neurons.size(); i++) {
            Neuron &neuron = output_layer.neurons[i];
            float error = expected_output[i] - neuron.output;
            neuron.delta = error * sigmoid_derivative(neuron.output);
        }

        // Backpropagate through hidden layers
        for (int l = layers.size() - 2; l >= 0; l--) {
            Layer &current_layer = layers[l];
            Layer &next_layer = layers[l + 1];

            for (int i = 0; i < current_layer.neurons.size(); i++) {
                Neuron &neuron = current_layer.neurons[i];
                float error = 0.0f;

                // Sum up the contributions of the next layer's deltas
                for (Neuron &next_neuron : next_layer.neurons) {
                    error += next_neuron.delta * next_neuron.weights[i];
                }

                neuron.delta = error * sigmoid_derivative(neuron.output);
            }
        }

        // Update all weights
        vector<float> input_to_current_layer = inputs;
        for (Layer &layer : layers) {
            for (Neuron &neuron : layer.neurons) {
                neuron.updateWeights(input_to_current_layer, learning_rate);
            }
            input_to_current_layer = layer.forward(input_to_current_layer); // Set the inputs for the next layer
        }
    }

    float computeLoss(const vector<float> &predicted_output, const vector<float> &expected_output) {
        float loss = 0.0f;
        for (int i = 0; i < predicted_output.size(); i++) {
            loss += pow(expected_output[i] - predicted_output[i], 2);
        }
        return loss / predicted_output.size();
    }

    void train(const vector<vector<float>> &training_data, const vector<vector<float>> &labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            float total_loss = 0.0f;
            for (int i = 0; i < training_data.size(); i++) {
                vector<float> predicted_output = forward(training_data[i]);
                total_loss += computeLoss(predicted_output, labels[i]);
                backpropagate(training_data[i], labels[i]);
            }
            cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / training_data.size() << endl;
        }
    }
};

void showVector(const vector<float> &v) {
    for (auto e : v) {
        cout << e << " ";
    }
    cout << endl;
}

int main() {
    vector<int> layers = {2, 3, 1}; // 2 inputs, 3 hidden neurons, 1 output
    NeuralNetwork network(layers, 0.1f);

    // Sample training data: XOR logic gate
    vector<vector<float>> training_data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<float>> labels = {{0}, {1}, {1}, {0}};

    network.train(training_data, labels, 10000);

    // Test the trained network
    for (const auto &input : training_data) {
        vector<float> output = network.forward(input);
        cout << "Input: ";
        showVector(input);
        cout << "Output: ";
        showVector(output);
    }
}
