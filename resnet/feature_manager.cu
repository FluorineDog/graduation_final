#include "memory_manager.h"
#include "engine.h"
using std::vector;
void FeatureManager::analyse(){
    const auto& graph = eng.forward_graph;
    vector<int> discover_time;
    {
        int time = 0;
        ProcedureDFS dfs(graph);
        
    }
}