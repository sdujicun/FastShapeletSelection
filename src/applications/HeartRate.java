/*
Data downloaded from Physionet. 
http://physionet.org/physiobank/database/meditation/
* 
* Peng C-K, Mietus JE, Liu Y, Khalsa G, Douglas PS, Benson H, Goldberger AL. Exaggerated Heart Rate Oscillations During Two Meditation Techniques. International Journal of Cardiology 70:101-107, 1999. 

There are five sets of observations
* 
•chi: 8 cases: Chi meditation group. There are two time series for each of the eight subjects (C1, C2, ... C8), denoted by record names with the suffix pre for the pre-meditation period and med for the meditation period. Each series is about one hour in duration.
 •yoga 4 cases: Kundalini Yoga meditation group. As for the Chi group, there are pre and med series for each of the four subjects (Y1, Y2, Y3, Y4). Durations range from 17 to 47 minutes.
 •normal 11 cases: Spontaneous breathing group (N1, N2, ... N11). Volunteers were recorded while sleeping. Durations are 6 hours each, except for N3 (4.6 hours).
 •metron 14 cases: Metronomic breathing group (M1, M2, ... M14). Volunteers were recorded while supine and breathing at a fixed rate of 0.25 Hz for 10 minutes.
 •ironman: Elite athletes (I1, I2, ... I9). Subjects participated in the Ironman Triathlon; the recordings were obtained during sleeping hours before the event. Durations range from 1 to 1.7 hours.
 
For TSC problem, we remove the pre cases for chi and do not use yoga, there being only 4 cases
*
* Series are in text format
* 
* C has between 3699 and 5090 obs
* Y has 910 to 1127 obs
* N1 17008 to 26923
* M has 469 to 770 
* I has 4200

* So problem 1 is going to be C, N and I, taking just the first 3600 readings
* 
* */
package applications;

import weka.core.elastic_distance_measures.DTW_DistanceBasic;
import weka.core.spectral_distance_functions.LikelihoodRatioDistance;
import fileIO.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.filters.*;
import weka.filters.timeseries.*;
import weka.filters.unsupervised.attribute.RemoveUseless;

public class HeartRate {
    public static String path="C:\\Research\\Data\\Time Series Classification\\HeartRate\\";
    public static DecimalFormat dc=new DecimalFormat("####.####");

    
	public static Classifier[] setSingleClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<Classifier>();
		Classifier c;
                PolyKernel kernel;

      /*		sc2.add(new IBk(1));
		names.add("NN");
          //      c=new DTW_1NN(1);
	//	((DTW_1NN)c).setMaxR(0.01);		
	//	sc2.add(c);
	//	names.add("NNDTW");
/*		sc2.add(new NaiveBayes());
		names.add("NB");
		sc2.add(new J48());
		names.add("C45");
*/		c=new SMO();
		kernel = new PolyKernel();
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
/*		c=new SMO();
		RBFKernel kernel2 = new RBFKernel();
		((SMO)c).setKernel(kernel2);
		sc2.add(c);
		names.add("SVMR");
		c=new RandomForest();
		((RandomForest)c).setNumTrees(30);
		sc2.add(c);
		names.add("RandF30");
		c=new RandomForest();
		((RandomForest)c).setNumTrees(100);
		sc2.add(c);
		names.add("RandF100");
		c=new RandomForest();
		((RandomForest)c).setNumTrees(500);
		sc2.add(c);
		names.add("RandF500");
		c=new RotationForest();
		sc2.add(c);
		names.add("RotF30");
*/	
		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);

		return sc;
	}

    public static void summariseData(){
        String[] prefixes={"C","C","N","I","M","Y","Y"};
        int [] counts={8,8,11,9,14,3,3};
        String[] suffixes={".med",".pre",".txt",".txt",".txt",".med",".pre"};
        InFile f;
        double t,h;
        for(int i=0;i<prefixes.length;i++){
            for(int j=1;j<=counts[i];j++){
                f= new InFile(path+prefixes[i]+j+suffixes[i]);
                System.out.println(prefixes[i]+j+suffixes[i]+" has "+f.countLines()+" cases");
            }
        }
     }   

    /** This method formats the series into Meditation/Non Meditation
     * and then forms the run distribution
     * The series are different length, so sh 
     */
    public static void formatArffHistos(){
        String[] prefixes={"C","C","N","I","M","Y","Y"};
        int [] counts={8,8,11,9,14,4,4};
        String[] suffixes={".med",".pre",".txt",".txt",".txt",".med",".pre"};
        InFile f;
        double t,h;
        int maxRuns=100;
        OutFile arff= new OutFile(path+"HeartRateHistos.arff");
        arff.writeLine("@RELATION HeartRate");
        for(int i=1;i<=maxRuns;i++)
            arff.writeLine("@ATTRIBUTE RunLength"+i+" real");
        arff.writeLine("@ATTRIBUTE PersonType {Med,NoMed}");
        arff.writeLine("@DATA");
        
        for(int i=0;i<prefixes.length;i++){
            for(int j=1;j<=counts[i];j++){
                f= new InFile(path+prefixes[i]+j+suffixes[i]);
                int c=f.countLines();
                f= new InFile(path+prefixes[i]+j+suffixes[i]);
                double[] d=new double[c];
                double mean=0;
                for(int k=0;k<c;k++){
                    f.readDouble();
                    d[k]=f.readDouble();
                    mean+=d[k];
                }
                mean/=c;
                System.out.println(" Mean ="+mean);
                //Normalise data
                for(int k=0;k<c;k++)
                    d[k]-=mean;
                RunLength rl = new RunLength();
                int[] hist=rl.processSingleSeries(d,maxRuns);
                //Normalise hist to proportions
                int sum=0;
                for(int k=0;k<hist.length;k++)
                    sum+=hist[k];
                System.out.println(" hist 1 = "+hist[0]+" sum ="+sum);
                for(int k=0;k<hist.length;k++)
                        arff.writeString((((double)hist[k])/sum)+",");
               if(i==0||i==5)
                        arff.writeLine("Med");
               else
                        arff.writeLine("NoMed");
            }
        }
     }   
        
    public static void formatArffACF(){
        String[] prefixes={"C","C","N","I","M","Y","Y"};
        int [] counts={8,8,11,9,14,4,4};
        String[] suffixes={".med",".pre",".txt",".txt",".txt",".med",".pre"};
        InFile f;
        double t,h;
        int maxRuns=100;
        OutFile arff= new OutFile(path+"HeartRateACF.arff");
        arff.writeLine("@RELATION HeartRate");
        for(int i=1;i<=maxRuns;i++)
            arff.writeLine("@ATTRIBUTE ACF"+i+" real");
        arff.writeLine("@ATTRIBUTE PersonType {Med,NoMed}");
        arff.writeLine("@DATA");
        
        for(int i=0;i<prefixes.length;i++){
            for(int j=1;j<=counts[i];j++){
                f= new InFile(path+prefixes[i]+j+suffixes[i]);
                int c=f.countLines();
                f= new InFile(path+prefixes[i]+j+suffixes[i]);
                double[] d=new double[c];
                double mean=0;
                for(int k=0;k<c;k++){
                    f.readDouble();
                    d[k]=f.readDouble();
                    mean+=d[k];
                }
                mean/=c;
                System.out.println(" Mean ="+mean);
                //Normalise data
                for(int k=0;k<c;k++)
                    d[k]-=mean;
                double[] acf=ACF.fitAutoCorrelations(d, 100);
                //Normalise hist to proportions
                int sum=0;
                for(int k=0;k<acf.length;k++)
                        arff.writeString(acf[k]+",");
               if(i==0||i==5)
                        arff.writeLine("Med");
               else
                        arff.writeLine("NoMed");
            }
        }
     }   
        
    public static void formatArff(){
        String[] prefixes={"C","N","I"};
        int [] counts={8,11,9};
        String[] suffixes={".med",".txt",".txt"};
        int nosLines=3600;
        OutFile arff= new OutFile(path+"HeartRateAll.arff");
        arff.writeLine("@RELATION HeartRate");
        for(int i=1;i<=nosLines;i++)
            arff.writeLine("@ATTRIBUTE t"+i+" real");
        arff.writeLine("@ATTRIBUTE PersonType {C,N,I}");
        arff.writeLine("@DATA");
        InFile f;
        double t,h;
        for(int i=0;i<prefixes.length;i++){
            for(int j=1;j<=counts[i];j++){
                f= new InFile(path+prefixes[i]+j+suffixes[i]);
                for(int k=0;k<nosLines;k++){
                    t=f.readDouble();
                    h=f.readDouble();
                    arff.writeString(h+",");
                }
                 arff.writeLine(prefixes[i]);
            }
        }
     }   
 
    public static void checkTimeIntervals(){
        String[] prefixes={"C","C","N","I","M","Y","Y"};
        int [] counts={8,8,11,9,14,4,4};
        String[] suffixes={".med",".pre",".txt",".txt",".txt",".med",".pre"};
        InFile f;
        OutFile of=new OutFile(path+"TimeDistribution.csv");
        for(int i=0;i<prefixes.length;i++){
            for(int j=1;j<=counts[i];j++){
                f= new InFile(path+prefixes[i]+j+suffixes[i]);
                int c=f.countLines();
                f= new InFile(path+prefixes[i]+j+suffixes[i]);
                double[] d=new double[c];
                double mean=0;
                double min=Double.MAX_VALUE;
                double max=0;
                d[0]=f.readDouble();
                f.readDouble();
                for(int k=1;k<c;k++){
                    d[k]=f.readDouble();
                    f.readDouble();
                    mean+=(d[k]-d[k-1]);
                    if(d[k]-d[k-1]>max)
                        max=d[k]-d[k-1];
                    if(d[k]-d[k-1]<min)
                        min=d[k]-d[k-1];
                }
                mean/=c;
                System.out.println(prefixes[i]+j+suffixes[i]+" Mean ="+mean+" Min ="+(min)+" Max ="+(max));
                of.writeLine(prefixes[i]+j+suffixes[i]+","+mean+","+(min)+","+(max));
            }
        }
    }
        
    
    
    public static void countLines(){
        String[] prefixes={"C","Y","N","M","I"};
        int [] counts={8,4,11,13,9};
        String[] suffixes={".med",".med",".txt",".txt",".txt"};
        InFile f;
        for(int i=0;i<prefixes.length;i++){
            for(int j=1;j<=counts[i];j++){
                f= new InFile(path+prefixes[i]+j+suffixes[i]);
                int c=f.countLines();
                System.out.println(prefixes[i]+j+suffixes[i]+" has "+c+" lines");
            }
        }
    }
    
/* Test classification with first four moments on the unnormalised data
 * 
 */
    public static void classifyOnSummaryStats(){
        Instances all=ClassifierTools.loadData(path+"HeartRate");
        SummaryStats s=new SummaryStats();
        try{
            Instances summary=s.process(all);
            Classifier c = new SMO();
            Evaluation e = new Evaluation(summary);
            e.crossValidateModel(c, summary, summary.numInstances(), new Random());
            double[][] c1=e.confusionMatrix();
            System.out.println(" Summary Stats Accuracy ="+(e.correct()/(double)summary.numInstances()));
            System.out.println(" Confusion Matrix =");
            for(int i=0;i<c1.length;i++){
                for(int j=0;j<c1[i].length;j++)
                    System.out.print(c1[i][j]+"\t");
                    System.out.print("\n");
            }
            
        }catch(Exception e){
            System.err.println(" Errrrr");
            System.exit(0);
        }    
    }
    

    public static void printConfusionMatrix(double[][] c1){
            System.out.println(" Confusion Matrix =");
            for(int i=0;i<c1.length;i++){
                for(int j=0;j<c1[i].length;j++)
                    System.out.print(c1[i][j]+"\t");
                    System.out.print("\n");
            }  
    }
    public static void classifyOnTransforms(Classifier c){
        Instances all=ClassifierTools.loadData(path+"HeartRate");
        SummaryStats s=new SummaryStats();
        int prop=50;
        int size=(all.numAttributes()-1)/prop;
        System.out.println(" TRUNCATE TO :"+size);
        try{
                System.out.println(" SUMMARY :");
                Instances trans=s.process(all);
                Evaluation e = new Evaluation(trans);
                e.crossValidateModel(c, trans, trans.numInstances(), new Random());
                System.out.println(" Summary Stats Accuracy ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                System.out.println(" RAW :");
                NormalizeCase nc = new NormalizeCase();
                nc.setNormType(NormalizeCase.NormType.STD_NORMAL);
                all=nc.process(all);
                OutFile of2=new OutFile(path+"HeartRateNormalised.arff");
                of2.writeString(all.toString());
                trans=new Instances(all);
                e=new Evaluation(trans);
                e.crossValidateModel(c, trans, trans.numInstances(), new Random());
                System.out.println(" RAW Accuracy ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                System.out.println(" ACF :");
                ACF acf = new ACF();
                acf.setMaxLag(size);
                trans=acf.process(all);
                e=new Evaluation(trans);
                e.crossValidateModel(c, trans, trans.numInstances(), new Random());
                System.out.println(" ACF Accuracy ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                System.out.println(" RUN LENGTHS :");
                RunLength rl= new RunLength();
                rl.setMaxRL(size);
                trans=rl.process(all);
                trans=nc.process(trans);
                e=new Evaluation(trans);
                e.crossValidateModel(c, trans, trans.numInstances(), new Random());
                System.out.println(" Run Length Accuracy 1NN Euclidean="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                e=new Evaluation(trans);
                kNN c2= new kNN(new DTW_DistanceBasic());
	//	((DTW_1NN)c).setMaxR(0.01);)
                e.crossValidateModel(c2, trans, trans.numInstances(), new Random());
                System.out.println(" Run Length Accuracy 1NN DTW ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                e=new Evaluation(trans);
                kNN c3= new kNN(new GowerDistance(trans));
	//	((DTW_1NN)c).setMaxR(0.01);)
                e.crossValidateModel(c3, trans, trans.numInstances(), new Random());
                System.out.println(" Run Length Accuracy GOWER ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                e=new Evaluation(trans);
                kNN c4= new kNN(new LikelihoodRatioDistance());
	//	((DTW_1NN)c).setMaxR(0.01);)
                e.crossValidateModel(c4, trans, trans.numInstances(), new Random());
                System.out.println(" Run Length Accuracy LR ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                System.out.println(" PACF :");
                PACF pacf = new PACF();
                pacf.setMaxLag(size);
                trans=pacf.process(all);
                e=new Evaluation(trans);
                e.crossValidateModel(c, trans, trans.numInstances(), new Random());
                System.out.println(" PACF Accuracy ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                System.out.println(" ARMA :");
                ARMA arma = new ARMA();
                arma.setMaxLag(size);
                trans=arma.process(all);
                e=new Evaluation(trans);
                e.crossValidateModel(c, trans, trans.numInstances(), new Random());
                System.out.println(" ARMA Accuracy ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
            
        }catch(Exception ex){
            System.err.println(" Errrrr: "+ex);
            System.exit(0);
        }    
    }

    public static void classifyFromHistos(){
        Instances data=ClassifierTools.loadData(path+"HeartRateAll");
        RemoveUseless filter=new RemoveUseless();
 
        try{
                filter.setInputFormat(data);
                for (int i = 0; i < data.numInstances(); i++) {
                    filter.input(data.instance(i));
                }
               filter.batchFinished();
                Instances trans = filter.getOutputFormat();
                Instance processed;
                while ((processed = filter.output()) != null) {
                    trans.add(processed);
                }
                System.out.println(" Histo size is="+trans.numAttributes());
                Classifier c=new kNN(1);
                System.out.println(" HISTO :");
                Evaluation e=new Evaluation(trans);
                e.crossValidateModel(c, trans, trans.numInstances(), new Random());
                System.out.println(" Euclid Accuracy ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                c=new kNN(new DTW_DistanceBasic());
                e=new Evaluation(trans);
                e.crossValidateModel(c, trans, trans.numInstances(), new Random());
                System.out.println(" DTW Accuracy ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                FastVector f=e.predictions();
                for(int i=0;i<f.size();i++)
                    System.out.println(" Pred "+i+" = "+f.elementAt(i));
                
                c=new kNN(new GowerDistance(trans));
                e=new Evaluation(trans);
               e.crossValidateModel(c, trans, trans.numInstances(), new Random());
                System.out.println(" Gower Euclidean="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
                c=new kNN(new LikelihoodRatioDistance(trans));

                e=new Evaluation(trans);
                e.crossValidateModel(c, trans, trans.numInstances(), new Random());
                System.out.println(" LR  ="+(e.correct()/(double)trans.numInstances()));
                printConfusionMatrix(e.confusionMatrix());
            
        }catch(Exception ex){
            System.err.println(" Errrrr: "+ex);
            System.exit(0);
        }    
    }
    
    
    public static void runlengthDebug(){
        Instances data=ClassifierTools.loadData(path+"HeartRateNormalised");
        try{
            RunLength rl= new RunLength();
            rl.setMaxRL(data.numAttributes()-1);
            Instances  trans=rl.process(data);
            OutFile of=new OutFile(path+"HeartRateRunLengths.csv");
 //           of.writeString(trans.toString());
            double[] series1=data.instance(0).toDoubleArray();
            double[] s=new double[series1.length-1];
            for(int i=0;i<s.length;i++)
                s[i]=series1[i];
            int [] hist=rl.processSingleSeries(s,s.length);
            for(int i=0;i<hist.length;i++)
                of.writeLine(hist[i]+","+trans.instance(0).value(i));
            
        }catch(Exception ex){
            System.err.println(" Errrrr: "+ex);
            System.exit(0);
        }    
    }
    public static void main(String[] args){
 //         checkTimeIntervals();
//        formatArffACF();
//        formatArffHistos();
//classifyFromHistos();
 //       runlengthDebug();
        summariseData();
//        classifyOnTransforms(new kNN(1));
 //       formatArff();
    }
    
    
}
