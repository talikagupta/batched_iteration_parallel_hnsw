#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <functional>
#include <memory>
#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <string>
#include <chrono>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "distance.h"

#define CUDA_CHECK(status) \
    if (status != 0) { \
        std::cerr << "CUDA Error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    }

using namespace std;

using NodeId = uint32_t;
using LayerId = int;

enum class InsertionPhase {
    DESCENDING_EF1,
    SEARCHING_EF_CONST,
    CONNECTING,
    DONE, 
    ERROR
};

// Calculate pairwise Squared L2 Distance
float calculateL2DistanceCPU(const float* vec1, const float* vec2, size_t dim) {

    if (!vec1 || !vec2) 
    {
        cout<<"invalid pointer"<<endl;
        return numeric_limits<float>::max();
    }
    
    float dist_sq = 0.0f;
    
    for (size_t i = 0; i < dim; ++i) {
        float diff = vec1[i] - vec2[i];
        dist_sq += diff * diff;
    }
    return dist_sq;
}


// Represents an element in priority queues (Distance, NodeId)
struct DistNode {
    float distance;
    NodeId nodeId;

    // For min-heap (Candidates C)
    bool operator>(const DistNode& other) const {
        return distance > other.distance;
    }
    // For max-heap (Found Neighbors W)
    bool operator<(const DistNode& other) const {
        return distance < other.distance;
    }
};

class HnswGraph {
    private:
        // --- Configuration & Parameters ---
        size_t vectorDim;       // Dimension of vectors
        int M;                  // Target number of connections (layers > 0)
        int maxM;               // Max connections per element per layer (>0) (Can be same as M)
        int maxM0;              // Max connections at layer 0
        float levelMultiplier;  // mL parameter
        size_t efConstruction;  // efConstruction value
    
        // --- Graph Structure ---
        // Layers stored bottom-up (index == layer ID)
        // Each layer maps NodeId -> vector of neighbor NodeIds
        vector<unordered_map<NodeId, vector<NodeId>>> layers;
        NodeId entryPoint = 0;  // Default entry point is node 0 (index)
        int maxLevel = -1;      // Highest layer index currently active (-1 means no layers)
    
        // --- Utilities ---
        std::mt19937 rng;       // Random number generator for level selection
    
        // --- Helper Methods ---
    
        // Ensure layer vector is large enough to access targetLayer
        void ensureLayerExists(LayerId targetLayer) {
            if (targetLayer < 0) return; // Ignore invalid layers
            // Layers stored bottom-up, index is layer ID
            if (static_cast<size_t>(targetLayer) >= layers.size()) {
                // Resize vector
                layers.resize(targetLayer + 1);
            }
            // Update maxLevel if this new layer is higher
            if (targetLayer > maxLevel) {
                maxLevel = targetLayer;
            }
        }
    
    public:
        
        // --- Data Storage ---
        const float* dataVectors = nullptr; // Pointer to external flat vector data (N x dim)
        size_t numVectors;              // Total number of vectors pointed to by dataVectors

        // --- Constructor ---
        HnswGraph(const float* data, size_t nVec, size_t dim,
                  int m_param, int maxM_param, int maxM0_param, float mL, size_t efC,
                  unsigned int seed = 100)
            : vectorDim(dim),
              M(m_param), maxM(maxM_param), maxM0(maxM0_param),
              levelMultiplier(mL), efConstruction(efC),
              dataVectors(data), numVectors(nVec),
              rng(seed) 
        {
            if (data == nullptr && nVec > 0) {
                throw runtime_error("HnswGraph created with null data pointer but non-zero vector count.");
            }
            
            cout << "HnswGraph Initialized: Nodes=" << numVectors << ", Dim=" << vectorDim
                       << ", M=" << M << ", maxM=" << maxM << ", maxM0=" << maxM0
                       << ", mL=" << levelMultiplier << ", efC=" << efConstruction << std::endl;
        }
    
        // --- Core Methods ---
    
        // Get pointer to vector data using NodeId as index
        const float* getVector(NodeId nodeId) const{
            if (dataVectors && nodeId < numVectors) {
                return dataVectors + (static_cast<size_t>(nodeId) * vectorDim);
            }
            // Return null if nodeId is out of bounds or data pointer is null
            cerr << "Warning: Attempted to get vector for out-of-bounds or null data node " << nodeId << endl;
            return nullptr;
        }
    
        // Get neighbors for a given node ID on a specific layer
         vector<NodeId>& getNeighbors(NodeId nodeId, LayerId layer) {
            static vector<NodeId> empty_neighbors; // Return ref to this if not found
            if (layer < 0 || static_cast<size_t>(layer) >= layers.size()) {
                return empty_neighbors; // Layer doesn't exist
            }
            // Access the map for the specific layer
            auto& layer_map = layers[layer];
            auto node_it = layer_map.find(nodeId);
            if (node_it != layer_map.end()) {
                return node_it->second; // Return reference to the vector of neighbors
            }
            return empty_neighbors; // Node not found in this layer map
        }
    
        // Add a bidirectional connection between two nodes on a specific layer
        void addConnections(NodeId node1, NodeId node2, LayerId layer) {
            ensureLayerExists(layer); // Make sure the layer vector is large enough
            if (layer < 0) return; // Cannot add to invalid layer
    
            // Check if nodes exist before adding connections
            if (!nodeExists(node1) || !nodeExists(node2)) {
                 cerr << "Warning: Attempting to add connection between non-existent nodes ("
                           << node1 << ", " << node2 << ") on layer " << layer << endl;
                 return;
            }
    
            if (node1 != node2) {
                // Add node2 to node1's list (avoid duplicates)
                if (find(layers[layer][node1].begin(), layers[layer][node1].end(), node2) == layers[layer][node1].end()){
                     layers[layer][node1].push_back(node2);
                }
                // Add node1 to node2's list
                if (find(layers[layer][node2].begin(), layers[layer][node2].end(), node1) == layers[layer][node2].end()){
                     layers[layer][node2].push_back(node1);
                }
            } else { // Handle self-connection (only for first node)
                 // Ensure node entry exists, add self only if list is empty
                 if (layers[layer].find(node1) == layers[layer].end() || layers[layer][node1].empty()) {
                      layers[layer][node1] = {node1}; // Create list with self
                 }
            }
        }
    
        // Set the neighbors for a node, replacing existing ones (used for shrinking)
        void setNeighbors(NodeId nodeId, LayerId layer, const std::vector<NodeId>& newNeighbors) {
            // Ensure layer exists, do nothing if layer is invalid
            ensureLayerExists(layer);
            if (layer < 0) {
                 std::cerr << "Warning: Attempted to set neighbors for node " << nodeId << " on invalid layer " << layer << std::endl;
                 return;
            }
            // Check if node exists before setting neighbors
            if (!nodeExists(nodeId)) {
                 std::cerr << "Warning: Attempted to set neighbors for non-existent node " << nodeId << " on layer " << layer << std::endl;
                 return;
            }
    
            // Assign the new neighbor list directly
            layers[layer][nodeId] = newNeighbors;
        }
    
        // --- Getters for Parameters and State ---
    
        size_t getDim() const { return vectorDim; }
        int getTopLevel() const { return maxLevel; }
        NodeId getEnterPoint() const { return entryPoint; }
        size_t getTotalNodes() const { return numVectors; } // Based on external data size
        size_t getEfConstruction() const { return efConstruction; } // Getter for efC
    
        // Get max allowed connections for a given layer
        size_t getMaxM(LayerId layer) const {
            // Use int comparison for safety as layer can be -1
            return (layer == 0) ? static_cast<size_t>(maxM0) : static_cast<size_t>(maxM);
        }
    
        // Check if NodeId is within the bounds of the external vector data
         bool nodeExists(NodeId nodeId) const {
            return nodeId < numVectors;
        }
    
        // --- Utilities ---
    
        // Calculate distance using NodeId as index into external data
        float calculateDistanceCPU(NodeId node1, NodeId node2) const {
            const float* vec1 = getVector(node1);
            const float* vec2 = getVector(node2);
            // calculateL2DistanceCPU handles null checks internally
            return calculateL2DistanceCPU(vec1, vec2, vectorDim);
        }
    
        // Generate random level
        int generateLevel() {
            if (levelMultiplier <= 0) return 0;
            uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
            float random_val = uniform_dist(rng);
            if (random_val == 0.0f) random_val = numeric_limits<float>::min();
            return static_cast<int>(-log(random_val) * levelMultiplier);
        }
    
        void setEnterPoint(NodeId nodeId) {
            if (nodeExists(nodeId)) {
                 entryPoint = nodeId;
            } else {
                std::cerr << "Warning: Attempted to set out-of-bounds node " << nodeId << " as enter point." << std::endl;
            }
        }
    
    };

// --- State for a Single Node Being Inserted ---
struct NodeInsertionState {
    NodeId nodeId;          // ID of the node being inserted
    HnswGraph& hnswGraph;
    size_t dim;
    InsertionPhase currentPhase;
    LayerId targetLevel;      // The randomly determined level 'l' for this node
    LayerId currentLayer;     // 'lc' - the layer currently being processed
    NodeId currentEnterPoint; // 'ep' for the current layer's search

    size_t currentEf;         // ef=1 or efConstruction

    // Data structures for the *current* SEARCH-LAYER execution
    
    priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>> candidates; // Min-heap C
    priority_queue<DistNode, std::vector<DistNode>, std::less<DistNode>> W;    // Max-heap W
    unordered_set<NodeId> visited;
    
    // std::vector<NodeId> visited_in_this_search; // To reset only visited nodes

    // Results from completed SEARCH_LAYER (for CONNECTING phase)
    vector<DistNode> searchResultsW; // Stores W after SEARCH_LAYER finishes

    // --- Batch Processing Helpers ---
    
    NodeId candidateToExplore = 0;  // The 'c' extracted from candidates
    vector<NodeId> neighborsToCalc; // Neighbors of 'c' needing distances

    NodeInsertionState(NodeId id, LayerId l, LayerId L, NodeId initialEp, size_t efConstructionValue, size_t dimension, HnswGraph& graph)
        : nodeId(id), dim(dimension),
          currentPhase(L > l ? InsertionPhase::DESCENDING_EF1 : InsertionPhase::SEARCHING_EF_CONST), // Initial Phase
          targetLevel(l), currentLayer(L), currentEnterPoint(graph.getEnterPoint()),
          currentEf(L > l ? 1 : efConstructionValue), // Set ef=1 if starting descent, 0 placeholder otherwise
          hnswGraph(graph)
           {
                // Initialize C, W, v for the first search step
                if (currentPhase == InsertionPhase::DESCENDING_EF1) {
                    currentEf = 1;
                } else {
                    // Directly start SEARCHING_EF_CONST phase for layer min(L,l)
                    currentLayer = std::min(L, targetLevel);
                    // efConstruction needs to be passed or accessible here
                    // currentEf = efConstruction; // Placeholder
                }
                // Adjust if graph was initially empty (L=-1)
                if (L == -1 && currentPhase == InsertionPhase::DESCENDING_EF1) {
                    currentPhase = InsertionPhase::SEARCHING_EF_CONST;
                    currentEf = efConstructionValue;
                    currentLayer = targetLevel; // Should be >= 0
                }
                resetSearchState(currentEnterPoint);
           }

    // Reset C, W, v for a new search on a layer
    void resetSearchState(NodeId ep) {
        // Clear queues and visited set
        while (!candidates.empty()) candidates.pop();
        while (!W.empty()) W.pop();
        visited.clear();

        // Add enter point(s)
        
        // Start with the single given enter point
        visited.insert(ep);

        float initialDistSq = calculateL2DistanceCPU(hnswGraph.getVector(nodeId), hnswGraph.getVector(ep), dim);

        // We need the distance(ep, q) - this requires an initial distance calc
        candidates.push({initialDistSq, ep}); // Placeholder distance, will be calculated
        W.push({initialDistSq, ep}); // Placeholder distance
        currentEnterPoint = ep;
        candidateToExplore = ep; // Start by exploring the enter point
        neighborsToCalc.clear(); // Will be populated in the first step
    }

    // Gets the furthest distance currently in W
    float getFurthestDistanceW() const {
        if (W.empty()) {
            return numeric_limits<float>::max();
        }
        return W.top().distance;
    }
};


// --- Batch Scheduler ---
class HnswBatchScheduler {
    private:
        HnswGraph& hnswGraph;
        size_t efConstruction;
        size_t MaxM; // a constant needed for allocating GPU memory, otherwise use hnswGraph.getMaxM(layer)
        size_t MaxM0;
        size_t batch_size;
    
        vector<unique_ptr<NodeInsertionState>> activeNodes;
        deque<NodeId> inputQueue; // Queue of node IDs waiting to be processed
    
        // GPU / Cublas resources
        cublasHandle_t cublasHandle;
        // PRE-ALLOCATED GPU Buffers
        float* d_x1_gpu = nullptr;         // For B x dim query vectors
        float* d_x2_gpu = nullptr;         // For B x MaxM x dim neighbor vectors
        float* d_output_gpu = nullptr;     // For B x MaxM output distances

        size_t dim;
    
        // Random number generation for levels
        std::mt19937 rng;
        std::uniform_real_distribution<double> unif{0.0, 1.0};
    
        // --- Helper Methods ---
    
    
        // Add waiting nodes to fill the batch
        void fillBatch() {
            while (activeNodes.size() < batch_size && !inputQueue.empty()) {
                NodeId nodeId = inputQueue.front();
            inputQueue.pop_front();

            LayerId l = hnswGraph.generateLevel();
            LayerId L = hnswGraph.getTopLevel(); // Current top layer
            NodeId ep = hnswGraph.getEnterPoint(); // Current enter point

            // Create a new NodeInsertionState with the correct parameters
            activeNodes.push_back(make_unique<NodeInsertionState>(
                nodeId, l, L, ep, efConstruction, dim, hnswGraph
            ));
                
            }
        }

        // Prepare data for GPU computation
        void prepareGpuData(size_t B, size_t R, float** h_x1, float** h_x2, float** h_output) {
            // Allocate temporary host buffers
            *h_x1 = new float[B * dim];  // x1: Query vectors (B x dim)
            *h_x2 = new float[B * R * dim]; // x2: Padded Neighbor vectors (B x R x M)
            *h_output = new float[B * R]; // output: Distances (B x R)
            
            // Create a dummy zero vector for padding
            vector<float> zeroVec(dim, 0.0f);

            // prepare matrix for query vectors
            for (size_t i = 0; i < B; ++i) {
                memcpy(*h_x1 + i * dim, hnswGraph.dataVectors + static_cast<size_t>(activeNodes[i]->nodeId) * dim, dim * sizeof(float));
            }

            // prepare matrix for neighbor vectors and pad if needed
            for (size_t i = 0; i < B; ++i) {
                NodeInsertionState& state = *activeNodes[i];
                size_t current_num_neighbors = state.neighborsToCalc.size();

                // Copy actual neighbors
                for (size_t j = 0; j < current_num_neighbors; ++j) {
                    NodeId neighborId = state.neighborsToCalc[j];
                    const float* neighborVec = hnswGraph.getVector(neighborId);
                    float* dest = *h_x2 + i * R * dim + j * dim; // Dest offset in h_x2
                    
                    memcpy(dest, neighborVec, dim * sizeof(float));
                }
                
                // Pad remaining slots with zeros
                for (size_t j = current_num_neighbors; j < R; ++j) {
                    float* dest = *h_x2 + i * R * dim + j * dim;
                    memcpy(dest, zeroVec.data(), dim * sizeof(float));
                }
            }
        }
    
        // The core synchronized step (one iteration of SEARCH-LAYER's while loop)
        void performSearchIteration() {
             
            size_t max_R = 0; // Maximum number of neighbors to calculate for any single node
            // 1. Prepare for Batch Distance Calculation
            for (auto& statePtr : activeNodes) {
                NodeInsertionState& state = *statePtr;
    
                // Extract nearest element c from C
                state.candidateToExplore = state.candidates.top().nodeId;
                state.candidates.pop();
    
                // Get neighbors of c
                const auto& neighbors = hnswGraph.getNeighbors(state.candidateToExplore, state.currentLayer);
    
                // Check if neighbor e is visited
                for (NodeId neighborId : neighbors) {
                    if (state.visited.find(neighborId) == state.visited.end()) {
                        state.neighborsToCalc.push_back(neighborId); //can have this as a set
                    }
                }

                // Update max_R for padding
                if (state.neighborsToCalc.size() > max_R) {
                    max_R = state.neighborsToCalc.size();
                }
            }

            if (max_R == 0) {
                // No distances to calculate in this iteration for any active node
                return;
            }
    
            // --- Stage 2: GPU Data Preparation (Padding) ---
            size_t B = activeNodes.size();
            size_t R = MaxM;      
            
            // host pointers
            float* h_x1 = nullptr;
            float* h_x2 = nullptr;
            float* h_output = nullptr;
            
            // Prepare data for GPU computation
            prepareGpuData(B, R, &h_x1, &h_x2, &h_output);

            // --- Stage 3: GPU Distance Calculation ---
            
            // Copy data to PRE-ALLOCATED GPU buffers
            CUDA_CHECK(cudaMemcpy(this->d_x1_gpu, h_x1, B * dim * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(this->d_x2_gpu, h_x2, B * R * dim * sizeof(float), cudaMemcpyHostToDevice));
            
            batched_L2(cublasHandle, d_x1_gpu, d_x2_gpu, d_output_gpu, B, R, dim);
           
            CUDA_CHECK(cudaMemcpy(h_output, this->d_output_gpu, B * R * sizeof(float), cudaMemcpyDeviceToHost));
    
            // 3. Update C, W, v for each node based on calculated distances
            for (size_t i = 0; i < activeNodes.size(); ++i) {
                NodeInsertionState& state = *activeNodes[i];

                for (size_t j = 0; j < state.neighborsToCalc.size(); ++j) {
                    NodeId neighborId = state.neighborsToCalc[j];
                    float dist_e_q = h_output[i * R + j];
                    state.visited.insert(neighborId);
                    if (dist_e_q < state.getFurthestDistanceW() || state.W.size() < state.currentEf) {
                        state.candidates.push({dist_e_q, neighborId});
                        state.W.push({dist_e_q, neighborId});
                        if (state.W.size() > state.currentEf) { state.W.pop(); }
                    }
                }
                state.neighborsToCalc.clear();
            }

            delete[] h_x1;
            delete[] h_x2;
            delete[] h_output;
        }
    
        // Select Neighbors (Algorithm 3)
        vector<NodeId> selectNeighborsSimple(NodeId q, const vector<DistNode>& W, size_t M) {
             // W comes from state.searchResultsW, which should be sorted by distance ascending
            vector<NodeId> selected;
            selected.reserve(M);
            size_t count = 0;
            for(const auto& dist_node : W) {
                 if (count++ >= M) break;
                 selected.push_back(dist_node.nodeId);
            }
            return selected;
        }

        vector<NodeId> selectNeighbors_without_sorted(NodeId q, const vector<NodeId>& C, size_t M, HnswGraph& hnswGraph) {
            // for shrinking connections

           // If candidate set size <= M, return all candidates
            if (C.size() <= M) {
                return C;
            }
            
            // Calculate distances from q to all candidates and store in pairs
            vector<pair<float, NodeId>> distances;
            distances.reserve(C.size());
            
            for (const NodeId& candidate : C) {
                float distance = hnswGraph.calculateDistanceCPU(q, candidate);
                distances.emplace_back(distance, candidate);
            }
            
            // Sort by distance (ascending)
            sort(distances.begin(), distances.end(), 
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            
            // Extract the M closest neighbors
            vector<NodeId> result;
            result.reserve(M);
            
            for (size_t i = 0; i < M && i < distances.size(); ++i) {
                result.push_back(distances[i].second);
            }
            
            return result;
       }
    
        // Process nodes that finished their SEARCH-LAYER or CONNECTING phase
        void handlePhaseTransitions() {
            for (int i = activeNodes.size() - 1; i >= 0; --i) { // Iterate backwards for safe removal
                NodeInsertionState& state = *activeNodes[i];
                bool nodeFinishedLayer = false;
    
                // Check if SEARCH-LAYER finished for this layer (C is empty or termination condition met)
                if ((state.currentPhase == InsertionPhase::DESCENDING_EF1 || state.currentPhase == InsertionPhase::SEARCHING_EF_CONST) ) {
                    float dist_c_q_sq = INT_MAX;
                    if(!state.candidates.empty())
                        dist_c_q_sq = state.candidates.top().distance;
                    
                    float furthestDistW_sq = state.getFurthestDistanceW();
                    if(state.candidates.empty() || dist_c_q_sq > furthestDistW_sq)
                     
                    {    
                        nodeFinishedLayer = true;
        
                        state.searchResultsW.clear();
                        while(!state.W.empty()){
                            state.searchResultsW.push_back(state.W.top());
                            state.W.pop();
                        }
        
                        reverse(state.searchResultsW.begin(), state.searchResultsW.end());

                    }
                    
                    // Sort W by distance ascending (priority queue was max-heap)
                    // std::sort(state.searchResultsW.begin(), state.searchResultsW.end(), [](const DistNode& a, const DistNode& b){
                    //    return a.distance < b.distance;
                    // });
    
                }
    
                if (nodeFinishedLayer) {
                    if (state.currentPhase == InsertionPhase::DESCENDING_EF1) {
                        // Finished ef=1 search for layer lc
    
                        // Get nearest from W as new ep
                        if (!state.searchResultsW.empty()) {
                            state.currentEnterPoint = state.searchResultsW[0].nodeId;
                        } else {
                            // Should not happen if ep was valid
                             state.currentPhase = InsertionPhase::ERROR;
                             continue;
                        }
    
                        state.currentLayer--; // Move to next layer down
    
                        if (state.currentLayer < state.targetLevel) {
                             // Should not happen based on loop logic (L..l+1)
                             // This node might be done if l was > L initially, handle this
                             state.currentPhase = InsertionPhase::DONE; // Or ERROR
                        } else if (state.currentLayer == state.targetLevel) {
                            // Transition to Phase 2
                            state.currentPhase = InsertionPhase::SEARCHING_EF_CONST;
                            state.currentEf = efConstruction;
                            state.resetSearchState(state.currentEnterPoint); // Reset C,W,v for efConstruction search
                        } else {
                            // Continue descending with ef=1
                            state.resetSearchState(state.currentEnterPoint); // Reset C,W,v for next ef=1 layer
                        }
    
                    } else if (state.currentPhase == InsertionPhase::SEARCHING_EF_CONST) {
                        // Finished ef=efConstruction search for layer lc, move to connecting
                        state.currentPhase = InsertionPhase::CONNECTING;
                        // Connection logic happens immediately below in the next phase check
                    }
                }
    
                 // --- Handle Connection Phase ---
                 if (state.currentPhase == InsertionPhase::CONNECTING) {
                     // Connect neighbors
                     size_t M_layer = hnswGraph.getMaxM(state.currentLayer);
                     vector<NodeId> neighbors = selectNeighborsSimple(state.nodeId, state.searchResultsW, M_layer);
                     if (neighbors.empty()) {
                        // this should never happen since W is initialized with the entry point
                        std::cerr << "Warning: Node " << state.nodeId << " has no connections at layer " << state.currentLayer << std::endl;
                    }                    
                     // Shrink neighbor connections if needed
                     for (NodeId neighborId : neighbors) {
                        // Add bidirectional connections
                        hnswGraph.addConnections(state.nodeId, neighborId, state.currentLayer);
                        
                        const auto& currentNeighbors = hnswGraph.getNeighbors(neighborId, state.currentLayer);
                        if (currentNeighbors.size() > hnswGraph.getMaxM(state.currentLayer)) {
                            // Need distances d(neighborId, its_neighbor) - using simple CPU version for this
                            size_t M_layer = hnswGraph.getMaxM(state.currentLayer);
                            vector<NodeId> newNeighbors = selectNeighbors_without_sorted(neighborId, currentNeighbors, M_layer, hnswGraph);
                            hnswGraph.setNeighbors(neighborId, state.currentLayer, newNeighbors); 
                        }
                     }
    
                     // Set ep for the next layer search from W
                     if (!state.searchResultsW.empty()) {
                        state.currentEnterPoint = state.searchResultsW[0].nodeId; // Use closest from W
                         
                     } else {
                        // If W is empty, keep the previous enter point? Or error?
                        // Keep currentEnterPoint as is for now.
                        cout << "Error, searchResultsW empty"<<endl;
                    }
    
                    state.currentLayer--; // Move to next layer down

                    if (state.currentLayer < 0) {
                        // Finished all layers for this node
                        state.currentPhase = InsertionPhase::DONE;
                    } else {
                        // Continue search on the next layer
                        state.currentPhase = InsertionPhase::SEARCHING_EF_CONST; // Still efConstruction
                        state.currentEf = efConstruction; // Ensure ef is set
                        state.resetSearchState(state.currentEnterPoint);
                    }
                 }
    
    
                // --- Handle DONE state ---
                if (state.currentPhase == InsertionPhase::DONE) {
                    // Update global enter point if needed
                    if (state.targetLevel > hnswGraph.getTopLevel()) {
                        hnswGraph.setEnterPoint(state.nodeId); // Thread-safe update needed
                    }
    
                    // Remove node from active batch
                    activeNodes.erase(activeNodes.begin() + i);
                } else if (state.currentPhase == InsertionPhase::ERROR) {
                    // Handle error
                    cerr << "Error processing node " << state.nodeId << std::endl;
                     
                    activeNodes.erase(activeNodes.begin() + i);
                }
            }
        }
    
    public:
        HnswBatchScheduler(HnswGraph& graph, size_t efConst, size_t batch_size, size_t MaxM)
            : hnswGraph(graph), efConstruction(efConst), batch_size(batch_size), rng(std::random_device{}()), MaxM(MaxM) {
    
            // Initialize Cublas
            cublasStatus_t status = cublasCreate(&cublasHandle);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("CUBLAS initialization failed!");
            }
            dim = hnswGraph.getDim();
    
            // Pre-allocate GPU memory
            CUDA_CHECK(cudaMalloc(&d_x1_gpu, batch_size * dim * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_x2_gpu, batch_size * MaxM * dim * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_output_gpu, batch_size * MaxM * sizeof(float)));
    
            activeNodes.reserve(batch_size);
        }
    
        ~HnswBatchScheduler() {
            // Cleanup Cublas
            cublasDestroy(cublasHandle);
            // Free pre-allocated GPU memory
            if (d_x1_gpu) cudaFree(d_x1_gpu);
            if (d_x2_gpu) cudaFree(d_x2_gpu);
            if (d_output_gpu) cudaFree(d_output_gpu);

            cout << "Scheduler Resources Cleaned Up." << endl;
        }

          // Add a new node to the scheduler
        void addNode(NodeId nodeId) {
            // Add the node directly to activeNodes if there's room, otherwise to inputQueue
            if (activeNodes.size() < batch_size) {
                // only initialize these parameters when you insert into the activenodes
                LayerId l = hnswGraph.generateLevel();
                LayerId L = hnswGraph.getTopLevel();
                NodeId ep = hnswGraph.getEnterPoint();
                
                activeNodes.push_back(make_unique<NodeInsertionState>(
                    nodeId, l, L, ep, efConstruction, dim, hnswGraph
                ));
            } else {
                inputQueue.push_back(nodeId);
            }
        }
    
        // Execute one synchronized step across the batch
        void step() {
            // 1. Refill batch from input queue if slots are available
            fillBatch();
    
            if (activeNodes.empty()) {
                return; // Nothing to process
            }
    
            // 2. Perform one iteration of the SEARCH-LAYER's while loop
            performSearchIteration();
    
            // 3. Handle phase transitions, connections, and node completion
            handlePhaseTransitions();
    
        }

        // Run until all nodes are processed
        void runUntilDone() {
            while (!activeNodes.empty() || !inputQueue.empty()) {             
                step();         
            }
        }

        // Check if there are any nodes being processed or waiting
        bool hasWork() const {
            return !activeNodes.empty() || !inputQueue.empty();
        }
    };