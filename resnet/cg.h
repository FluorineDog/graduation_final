

class ForwardVisitor : public Visitor {
  public:
    void start(Engine* eng, const float* input, float* logits){
        
    }
    virtual void visit(FCNode&) override{
        
    }
    virtual void visit(ActivationNode&) override{
        
    }
    Engine* eng; 
};