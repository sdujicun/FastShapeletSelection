/*
Code to generate results for the paper
Lines, Jason and Bagnall, Anthony (2014) Time series classification with 
ensembles of elastic distance measures. Data Mining and Knowledge Discovery Journal

also published in 
Lines, Jason and Bagnall, Anthony (2014) Ensembles of Elastic Distance 
easures for Time Series Classification. In: Proceedings of SDM 2014
 */
package papers;

import tsc_algorithms.ElasticEnsemble;
import development.DataSets;
import fileIO.OutFile;
import java.text.DecimalFormat;
import utilities.ClassifierTools;
import weka.classifiers.meta.timeseriesensembles.*;
import weka.core.Instances;


public class DAMI2014_Lines {
    //CHANGE THIS
    static String path="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";

/** This will be fairly slow! The EE is threaded for cross validation,but
 this also makes it memory hungry.
 
 * In reality, we decomposed the ensemble components and ran them concurrently. 
 * However, this approach is clearer, it does work and is sufficicent for small 
 * problems
 */    
    public static double singleProblem(String problem, ElasticEnsemble.EnsembleType e) throws Exception{
        ElasticEnsemble ee= new ElasticEnsemble();
        ee.setEnsembleType(e);
        Instances train = ClassifierTools.loadData(path+problem+"\\"+problem+"_TRAIN");
        ee.buildClassifier(train);
        Instances test = ClassifierTools.loadData(path+problem+"\\"+problem+"_TEST");
        double a=ClassifierTools.accuracy(test, ee);
        return a;
    }
    
    public static double[] smallUCRProblems() throws Exception{
        OutFile of = new OutFile(path+"SmallUCRProblems.csv"); 
        double[] acc = new double[DataSets.ucrSmall.length];
        DecimalFormat df = new DecimalFormat("##.###");
        for(int i=0;i<DataSets.ucrSmall.length;i++){
            acc[i]=singleProblem(DataSets.ucrSmall[i],ElasticEnsemble.EnsembleType.Equal);
            System.out.println(DataSets.ucrSmall[i]+" Error = "+df.format(1-acc[i]));
        }
        return acc;
    }
/* All problems: with no CV
    */
    public static double[] allProblems() throws Exception{
        OutFile of = new OutFile(path+"SmallUCRProblems.csv"); 
        double[] acc = new double[DataSets.ucrSmall.length];
        DecimalFormat df = new DecimalFormat("##.###");
        for(int i=0;i<DataSets.fileNames.length;i++){
            acc[i]=singleProblem(DataSets.fileNames[i],ElasticEnsemble.EnsembleType.Equal);
            System.out.println(DataSets.fileNames[i]+" Error = "+df.format(1-acc[i]));
            of.writeLine(DataSets.fileNames[i]+","+(1-acc[i]));
        }
        return acc;
    }
    
    
    public static void main(String[] args) throws Exception{
/* Single Problem
        String prob="ItalyPowerDemand";
        double a=singleProblem(prob);
        System.out.println(" EE Train/Test Acc = "+a);
*/
 //       smallUCRProblems();
        allProblems();
    }
    
    
}
