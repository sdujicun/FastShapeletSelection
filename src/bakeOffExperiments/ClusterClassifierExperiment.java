/*

 */
package bakeOffExperiments;

import development.DataSets;
import fileIO.OutFile;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * @author ajb
 * 
 * conducts 100 resample experiments sequentially
 * Results are written to the file passed in the form
 * 
 * 
 */
public class ClusterClassifierExperiment {
    public static String resultsPath=DataSets.clusterPath+"Results/";
    
//Store and save accuracies and predictions    
    public static double[] resampleExperiment(Instances train, Instances test, Classifier c, int resamples,OutFile of,String preds){

       double[] foldAcc=new double[resamples];
        for(int i=0;i<resamples;i++){
            Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);
            double act,pred;
            OutFile p=new OutFile(preds+"/fold"+i+".csv");
            try{              
                c.buildClassifier(data[0]);
                foldAcc[i]=0;
                for(int j=0;j<data[1].numInstances();j++)
                {
                    act=data[1].instance(j).classValue();
                    pred=c.classifyInstance(data[1].instance(j));
                    if(act==pred)
                        foldAcc[i]++;
                    p.writeLine(act+","+pred);
                }
                foldAcc[i]/=data[1].numInstances();
                of.writeString(foldAcc[i]+",");

            }catch(Exception e)
            {
                    System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                    e.printStackTrace();
                    System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                    System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                    System.exit(0);
            }
        }            
         return foldAcc;
    }
}
