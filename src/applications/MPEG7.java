/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package applications;

import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import fileIO.OutFile;
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.*;
import weka.filters.timeseries.*;
import weka.core.shapelet.*;

/**
 *
 * @author classification o0f image outlines froom MPEG 7 standard
 */

public class MPEG7 {

    public static String[] fileNames ={"MPEG7256_20Shapes",
        "MPEG7512_20Shapes",
        "MPEG71024_20Shapes"
    };
    public static String[] xy_fileNames ={"OUTLINE MPEG7256_20Shapes",
        "MPEG7512_20Shapes",
        "MPEG71024_20Shapes"
    };
    
    public static Classifier[] outlineSeriesTransformClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<Classifier>();
		Classifier c;
                
                IBk c2=new IBk(1);
                c2.setCrossValidate(false);
                sc2.add(c2);
		names.add("OneNN");
                c2=new IBk(40);
                c2.setCrossValidate(true);
                sc2.add(c2);
		names.add("kNN");
                c=new DTW_1NN();
		sc2.add(c);
		names.add("OneNNDTW_10");
 /*          
		sc2.add(new NaiveBayes());
		names.add("NB");
		sc2.add(new J48());
		names.add("C45");
		c=new SMO();
		PolyKernel kernel = new PolyKernel();
		kernel.setExponent(1);
		((SMO)c).setKernel(kernel);
		sc2.add(c);
		names.add("SVML");
		c=new SMO();
		kernel = new PolyKernel();
		kernel.setExponent(2);
		((SMO)c).setKernel(kernel);
		sc2.add(c);
		names.add("SVMQ");
		c=new SMO();
//		RBFKernel kernel2 = new RBFKernel();
//		((SMO)c).setKernel(kernel2);
//		sc2.add(c);
//		names.add("SVMR");
//		c=new RandomForest();
//		((RandomForest)c).setNumTrees(30);
//		sc2.add(c);
//		names.add("RandF30");
		c=new RandomForest();
		((RandomForest)c).setNumTrees(100);
		sc2.add(c);
		names.add("RandF100");
		c=new RotationForest();
		sc2.add(c);
		names.add("RotF30");
*/
	
		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);

		return sc;
	}
    
    public static Classifier[] outlineDataSetTransformClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<Classifier>();
		Classifier c;
                IBk c2=new IBk(1);
                c2.setCrossValidate(false);
                sc2.add(c2);
		names.add("OneNN");
                c2=new IBk(40);
                c2.setCrossValidate(true);
                sc2.add(c2);
		names.add("kNN");
		sc2.add(new NaiveBayes());
		names.add("NB");
		sc2.add(new J48());
		names.add("C45");
		c=new SMO();
		PolyKernel kernel = new PolyKernel();
		kernel.setExponent(1);
		((SMO)c).setKernel(kernel);
		sc2.add(c);
		names.add("SVML");
		c=new SMO();
		kernel = new PolyKernel();
		kernel.setExponent(2);
		((SMO)c).setKernel(kernel);
		sc2.add(c);
		names.add("SVMQ");
		c=new SMO();
//		RBFKernel kernel2 = new RBFKernel();
//		((SMO)c).setKernel(kernel2);
//		sc2.add(c);
//		names.add("SVMR");
//		c=new RandomForest();
//		((RandomForest)c).setNumTrees(30);
//		sc2.add(c);
//		names.add("RandF30");
		c=new RandomForest();
		((RandomForest)c).setNumTrees(100);
		sc2.add(c);
		names.add("RandF100");
		c=new RotationForest();
		sc2.add(c);
		names.add("RotF30");

	
		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);

		return sc;
	}
    

   public static void XYClassifier(){
        
         ArrayList<String> names=new ArrayList<String>();
         Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\MPEG7Shapes\\OUTLINE MPEG7256_20Shapes");
        all.randomize(new Random());
        Classifier[] c =outlineDataSetTransformClassifiers(names);
        OutFile res=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\MPEG7\\XY_PCA_256.csv");
        try{
		System.out.println("\n******************PCA Domain******************");
//So we need to do this so the model is not fitted on the training data
                double[] correct=new double[c.length];
                for(int i=0;i<all.numInstances();i++){
                    System.out.println("Instance ="+i);
                    PrincipalComponents pca=new PrincipalComponents (); 
                    Instances train=new Instances(all);
                    Instances test=new Instances(all,i,1);
                    train.delete(i);
                    pca.buildEvaluator(train);
                    Instances pcaTrain=pca.transformedData(train);
                    Instances pcaTest=pca.transformedData(test);
                    for(int j=0;j<c.length;j++){
                        c[j].buildClassifier(pcaTrain);
                        double pred=c[j].classifyInstance(pcaTest.instance(0));
                        if(pred==all.instance(i).classValue())
                            correct[j]++;
                    }
                }
                for(int j=0;j<c.length;j++){
                    correct[j]/=all.numInstances();
                    res.writeLine(names.get(j)+","+correct[j]);
                    System.out.println(names.get(j)+","+correct[j]);
                }
        }catch(Exception ex){
                ex.printStackTrace();
                System.exit(0);
        }       
    }
// Raw, Spectrum and ACF    
    // PCA outline, Shapelets
    public static void outineDataSetTransformClassifier(String file, String transform){
        
         ArrayList<String> names=new ArrayList<String>();
//        Instances all=ClassifierTools.loadData("C:\\Research\\Data\\Time Series Classification\\Otoliths\\Herring2");
//        Instances all=ClassifierTools.loadData("C:\\Research\\Data\\Time Series Classification\\Otoliths\\HerringPlaice");
         Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\MPEG7Shapes\\"+file);
        all.randomize(new Random());
        int folds=all.numInstances();
        NormalizeCase nc=new NormalizeCase();
        nc.setNormType(NormalizeCase.NormType.STD);
        Classifier[] c =outlineDataSetTransformClassifiers(names);
        OutFile res=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\MPEG7\\"+transform+"_"+file+".csv");
        try{
            all=nc.process(all);
            if(transform.equals("PCA")){
		System.out.println("\n******************PCS Domain******************");
//So we need to do this so the model is not fitted on the training data
                double[] correct=new double[c.length];
                for(int i=0;i<all.numInstances();i++){
                    System.out.println("Instance ="+i);
                    PrincipalComponents pca=new PrincipalComponents (); 
                    Instances train=new Instances(all);
                    Instances test=new Instances(all,i,1);
                    train.delete(i);
                    pca.buildEvaluator(train);
                    Instances pcaTrain=pca.transformedData(train);
                    Instances pcaTest=pca.transformedData(test);
                    for(int j=0;j<c.length;j++){
                        c[j].buildClassifier(pcaTrain);
                        double pred=c[j].classifyInstance(pcaTest.instance(0));
                        if(pred==all.instance(i).classValue())
                            correct[j]++;
                    }
                }
                for(int j=0;j<c.length;j++){
                    correct[j]/=all.numInstances();
                res.writeLine(names.get(j)+","+correct[j]);
                System.out.println(names.get(j)+","+correct[j]);
               
                }
            }
           else if(transform.equals("Shapelet")){   //Bias if we do it on the whole data?
 
		System.out.println("\n******************SHAPELET Domain******************");
//So we need to do this so the model is not fitted on the training data
                double[] correct=new double[c.length];
                for(int i=0;i<all.numInstances();i++){
                    System.out.println("Instance ="+i);
                    FullShapeletTransform shape=new FullShapeletTransform(100,100,100); 
                    Instances train=new Instances(all);
                    Instances test=new Instances(all,i,1);
                    train.delete(i);
                    
                    Instances shapeletTrain=shape.process(train);
                    Instances shapeletTest=shape.process(test);
                    for(int j=0;j<c.length;j++){
                        c[j].buildClassifier(shapeletTrain);
                        double pred=c[j].classifyInstance(shapeletTest.instance(0));
                        if(pred==all.instance(i).classValue())
                            correct[j]++;
                    }
                }
                for(int j=0;j<c.length;j++){
                    correct[j]/=all.numInstances();
                res.writeLine(names.get(j)+","+correct[j]);
                System.out.println(names.get(j)+","+correct[j]);
               
                }
            }
        }catch(Exception ex){
                ex.printStackTrace();
                System.exit(0);
        }       
    }
// Raw, Spectrum and ACF
    public static void outlineSeriesTransformClassifier(String file, String transform){
        ArrayList<String> names=new ArrayList<String>();
//        Instances all=ClassifierTools.loadData("C:\\Research\\Data\\Time Series Classification\\Otoliths\\Herring2");
//        Instances all=ClassifierTools.loadData("C:\\Research\\Data\\Time Series Classification\\Otoliths\\HerringPlaice");
         Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\MPEG7Shapes\\"+file);
        all.randomize(new Random());
        int folds=all.numInstances();
        NormalizeCase nc=new NormalizeCase();
        nc.setNormType(NormalizeCase.NormType.STD);
        Classifier[] c =outlineSeriesTransformClassifiers(names);
        OutFile res=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\MPEG7\\"+transform+"_"+file+".csv");
        try{
            all=nc.process(all);
            if(transform.equals("ACF")){
                ACF acf=new ACF();
                acf.setMaxLag(all.numAttributes()/5);
                all=acf.process(all);
            }
            else if(transform.equals("PowerSpectrum")){
                PowerSpectrum ps=new PowerSpectrum();
                all=ps.process(all);
            }
/*            else if(transform.equals("Shapelet")){   //Bias if we do it on the whole data?
                Shapelet shape=new Shapelet(all.numAttributes()/2,3,all.numAttributes()/5);
                all=shape.process(all);

            }
            else if(transform.equals("PCA")){   //Bias if we do it on the whole data?

            }
*/            for(int i=0;i<c.length;i++){
                System.out.print(" running classifier "+names.get(i));
                Evaluation e=new Evaluation(all);
                e.crossValidateModel(c[i], all, folds, new Random());
                res.writeLine(names.get(i)+","+e.correct()/all.numInstances());
                System.out.println(" Acc = "+e.correct()/all.numInstances());
            }
        }catch(Exception ex){
                ex.printStackTrace();
                System.exit(0);
        }
    }
	    public static void main(String[] args){
//                CSSClassifier(100);
                XYClassifier();
                outineDataSetTransformClassifier(fileNames[0],"Shapelet");
//                   outlinesClassifier(fileNames[1],"PowerSpectrum");
//                for(int i=1;i<fileNames.length;i++){
 //                  outlineSeriesTransformClassifier(fileNames[i],"Raw");
//                }
//                   outlinesClassifier(fileNames[2],"Raw");
//                for(String s:fileNames)
//                   rawOutlinesClassifier(s,"Shapelet");

                
                
                //        basicDataTransforms("NB",10);
 //       basicDataTransforms("SVMQ",10);
 //       basicDataTransforms("NNDTW",10);
        
    }

    
}
