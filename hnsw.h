#ifndef HNSW_H
#define HNSW_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <cstdint>
#include <cublas_v2.h>  


using NodeId = uint32_t;
using LayerId = int;

enum class InsertionPhase {
    DESCENDING_EF1,
    SEARCHING_EF_CONST,
    CONNECTING,
    DONE,
    ERROR
};

struct DistNode {
    float distance;
    NodeId nodeId;
    bool operator>(const DistNode& other) const;
    bool operator<(const DistNode& other) const;
};

class HnswGraph {
public:
    HnswGraph(const float* data, size_t nVec, size_t dim,
              int m_param, int maxM_param, int maxM0_param, float mL, size_t efC,
              unsigned int seed = 100);

    const float* getVector(NodeId nodeId) const;
    std::vector<NodeId>& getNeighbors(NodeId nodeId, LayerId layer);
    void addConnections(NodeId node1, NodeId node2, LayerId layer);
    void setNeighbors(NodeId nodeId, LayerId layer, const std::vector<NodeId>& newNeighbors);

    size_t getDim() const;
    int getTopLevel() const;
    NodeId getEnterPoint() const;
    size_t getTotalNodes() const;
    size_t getEfConstruction() const;
    size_t getMaxM(LayerId layer) const;
    bool nodeExists(NodeId nodeId) const;
    float calculateDistanceCPU(NodeId node1, NodeId node2) const;
    void setEnterPoint(NodeId nodeId);

private:
    size_t vectorDim;
    int M, maxM, maxM0;
    float levelMultiplier;
    size_t efConstruction;

    std::vector<std::unordered_map<NodeId, std::vector<NodeId>>> layers;
    NodeId entryPoint = 0;
    int maxLevel = -1;
    std::mt19937 rng;

    void ensureLayerExists(LayerId targetLayer);

public:
    const float* dataVectors = nullptr;
    size_t numVectors;

    int generateLevel();
};

struct NodeInsertionState {
    NodeId nodeId;
    HnswGraph& hnswGraph;
    size_t dim;
    InsertionPhase currentPhase;
    LayerId targetLevel, currentLayer;
    NodeId currentEnterPoint;
    size_t currentEf;

    std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>> candidates;
    std::priority_queue<DistNode, std::vector<DistNode>, std::less<DistNode>> W;
    std::unordered_set<NodeId> visited;
    std::vector<DistNode> searchResultsW;

    NodeId candidateToExplore = 0;
    std::vector<NodeId> neighborsToCalc;

    NodeInsertionState(NodeId id, LayerId l, LayerId L, NodeId initialEp,
                       size_t efConstructionValue, size_t dimension, HnswGraph& graph);
    void resetSearchState(NodeId ep);
    float getFurthestDistanceW() const;
};

// Utility function
float calculateL2DistanceCPU(const float* vec1, const float* vec2, size_t dim);

class HnswBatchScheduler {
    private:
        HnswGraph& hnswGraph;
        size_t efConstruction;
        size_t MaxM;
        size_t MaxM0;
        size_t batch_size;
    
        std::vector<std::unique_ptr<NodeInsertionState>> activeNodes;
        std::deque<NodeId> inputQueue;
    
        // GPU resources
        cublasHandle_t cublasHandle;
        float* d_x1_gpu = nullptr;
        float* d_x2_gpu = nullptr;
        float* d_output_gpu = nullptr;
    
        size_t dim;
    
        std::mt19937 rng;
        std::uniform_real_distribution<double> unif{0.0, 1.0};
    
        // Internal helper functions
        void fillBatch();
        void prepareGpuData(size_t B, size_t R, float** h_x1, float** h_x2, float** h_output);
        void performSearchIteration();
        std::vector<NodeId> selectNeighborsSimple(NodeId q, const std::vector<DistNode>& W, size_t M);
        std::vector<NodeId> selectNeighbors_without_sorted(NodeId q, const std::vector<NodeId>& C, size_t M, HnswGraph& hnswGraph);
        void handlePhaseTransitions();
    
    public:
        HnswBatchScheduler(HnswGraph& graph, size_t efConst, size_t batch_size, size_t MaxM);
        ~HnswBatchScheduler();
    
        void addNode(NodeId nodeId);
        void step();
        void runUntilDone();
        bool hasWork() const;
    };


#endif // HNSW_H
