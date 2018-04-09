package weka.classifiers.meta.timeseriesensembles;
import java.util.ArrayList;
import java.util.Random;
import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.NormalizeAttribute;
import weka.filters.NormalizeCase;
import weka.filters.SimpleBatchFilter;
import weka.filters.timeseries.*;

public class SingleTransformEnsembles extends AbstractClassifier{


    enum TransformType {TIME,PS,ACF}; 
    TransformType t = TransformType.TIME;
    SimpleBatchFilter transform;
    Classifier[] classifiers;
    Instances train;

    public SingleTransformEnsembles(){
        super();
        initialise();
    }
    public final void initialise(){
//Transform            
        switch(t){
            case TIME:
                transform=new NormalizeCase();
                break;
                

        }

    }
    @Override
    public void buildClassifier(Instances data){
        
    }        
  
	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}
	public static void main(String[] args){
//Load up Beefand test only on that
		
		
	}
	
}
