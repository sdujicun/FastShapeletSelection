/*
Wrapper for SMO which sets 
 */
package weka.classifiers.functions;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.core.*;

/**
 *
 * @author ajb
 */
public class OptimisedSVM extends SMO{
    int min=-12;
    int max=6;
    private static int MAX_FOLDS=100;
    @Override
    public void buildClassifier(Instances train) throws Exception {
        double c=Math.pow(2,min);
        int folds=(train.numInstances()>MAX_FOLDS)?MAX_FOLDS:train.numInstances();
        double minErr=1;
        double bestC=c;
        for(int i=0;i<=(max-min);i++){
            Evaluation eval;
            SMO model = new SMO();
            model.setKernel(m_kernel);
            model.setC(c);
            eval = new Evaluation(train);
            eval.crossValidateModel(model, train, folds,new Random());
            double e=eval.errorRate();
            System.out.println(" c ="+c+" error ="+e);
            if(e<minErr){
                e=minErr;
                bestC=c;
            }
            c*=2;
        }
        setC(bestC);
        super.buildClassifier(train);
    }
}
