/*
WORMS
 */

package applications;

import applications.HandandBoneOutlines;
import fileIO.InFile;
import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.SummaryStats;
import weka.filters.timeseries.shapelet_transforms.*;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.OnlineSubSeqDistance;

/**
 *
 * @author ajb
 */
public class Worms {
    static int[] seriesLengths;
    static String[] classLabels={"N2","goa1", "unc1","unc38","unc63"}; 
    static int nosSeries;
    static DecimalFormat df= new DecimalFormat("##.####");
    static String path="C:\\Users\\ajb\\Dropbox\\Worms Project\\";
    static int newLength=900; 
    public static void formatArff(){
//Load into array
        InFile f= new InFile("C:\\Users\\ajb\\Dropbox\\Worms Project\\dataSeries.csv");
        InFile f2= new InFile("C:\\Users\\ajb\\Dropbox\\Worms Project\\seriesLengths.csv");
        nosSeries=f2.countLines();
        double[][] series=new double[nosSeries][];
        String[] labels=new String[nosSeries];
        seriesLengths=new int[nosSeries];
        System.out.println(" Number of series in new data set ="+nosSeries);
       
        f2= new InFile("C:\\Users\\ajb\\Dropbox\\Worms Project\\seriesLengths.csv");
        for(int i=0;i<nosSeries;i++){
            seriesLengths[i]=f2.readInt();
            series[i]=new double[seriesLengths[i]];
            for(int j=0;j<seriesLengths[i];j++){
                series[i][j]=f.readDouble();
            }
            labels[i]=f.readString();
        }
        System.out.println(" Finished Loading");
//Downsample to length 900 by averaging over the second        
        double[][] newSeries=new double[nosSeries][newLength];
        for(int i=0;i<nosSeries;i++){
            int count =0;
            int period=newSeries[i].length/newLength;
            for(int j=0;j<newSeries[i].length;j++){
                newSeries[i][j]=series[i][count++];
                for(int k=1;k<period;k++)
                    newSeries[i][j]+=series[i][count++];
                newSeries[i][j]/=period;    
            }
        }
//Write to Arff =
        OutFile out = new OutFile("C:\\Users\\ajb\\Dropbox\\Worms Project\\TS-data-only\\worms.arff");
        out.writeLine("@Relation worms");
        for(int j=0;j<newSeries[0].length;j++)
        out.writeLine("@attribute eigenWorm1_"+(j+1)+" numeric");
        out.writeString("@attribute type {");
        for(int j=0;j<classLabels.length-1;j++)
            out.writeString(classLabels[j]+",");
        out.writeLine(classLabels[classLabels.length-1]+"}");
        out.writeLine("\n@data\n");
        int[] classCounts=new int[classLabels.length];
        for(int i=0;i<nosSeries;i++){
            for(int j=0;j<newSeries[i].length;j++){
                out.writeString(df.format(newSeries[i][j])+",");
            }
            labels[i]=labels[i].trim();
            if(labels[i].equals("goa-1"))
                System.out.println(labels[i]);
            switch(labels[i]){
                case "N2":
                    out.writeString("N2");
                    classCounts[0]++;
                    break;
                case "goa-1":
                    out.writeString("goa1");
                    classCounts[1]++;
                    break;
                case "unc-1":
                    out.writeString("unc1");
                    classCounts[2]++;
                    break;
                case "unc-38":
                    out.writeString("unc38");
                    classCounts[3]++;
                    break;
                case "unc-63":
                    out.writeString("unc63");
                    classCounts[4]++;
                    break;
            }
                    out.writeString("\n");
            
        }
        for(int i=0;i<classCounts.length;i++)
            System.out.println(" Label "+classLabels[i]+" has "+classCounts[i]+" cases");
/*
         Label goa1 has 109 cases
         Label N2 has 44 cases
         Label unc1 has 35 cases
 Label unc38 has 45 cases
 Label unc63 has 25 cases
        */            
    }
    
    public static void makeTestTrainSplit(){
        int[] overallSize={109,44,35,45,25};
        int[] testSize={33,13,10,13,8};
        Instances all=ClassifierTools.loadData(path+"worms");
        Instances train=new Instances(all);
        Instances test=new Instances(all,0);
        int pos=0;
        for(int i=0;i<testSize.length;i++){
            System.out.println(" Extracting class "+i);
            for(int j=0;j<testSize[i];j++){//Find 
               if(train.instance(pos).classValue()==i){
                   test.add(train.instance(pos));
                   train.delete(pos);
                   System.out.println("moving "+pos+" class "+i);
               }else if(pos<train.numInstances()){
                   j--;
                   pos++;
                   while(pos<train.numInstances()&& train.instance(pos).classValue()!=i)
                       pos++;
               }
            }
            OutFile trainF=new OutFile(path+"worms_TRAIN.arff");
            trainF.writeString(train.toString());
            OutFile testF=new OutFile(path+"worms_TEST.arff");
            testF.writeString(test.toString());
        }
    }
    
    
    public static void basicShapeletTestTrain(){
        Instances train=ClassifierTools.loadData(path+"worms_TRAIN");
        Instances test=ClassifierTools.loadData(path+"worms_TEST");
        NormalizeCase nc = new NormalizeCase();
        try {
            train=nc.process(train);
            test=nc.process(test);
        } catch (Exception ex) {
            Logger.getLogger(Worms.class.getName()).log(Level.SEVERE, null, ex);
        }
        int nosShapelets=2000;
        FullShapeletTransform st = new FullShapeletTransform();
        st.setSubSeqDistance(new OnlineSubSeqDistance());
        st.setNumberOfShapelets(nosShapelets);
        int minLength=4;
        int maxLength=200;
        st.setShapeletMinAndMax(minLength, maxLength);
        st.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.F_STAT);        
        Instances shapeletTrain=null;
        Instances shapeletTest=null;

        try {
            shapeletTrain=st.process(train);
            shapeletTest=st.process(test);
        } catch (Exception ex) {
            System.out.println("Error performing the shapelet transform"+ex);
            ex.printStackTrace();
            System.exit(0);
        }
        System.out.println(" Transform is complete");
        OutFile of = new OutFile(path+"shapeletWorms_TRAIN");
        of.writeString(shapeletTrain.toString());
        of = new OutFile(path+"shapeletWorms_TEST");
        of.writeString(shapeletTest.toString());
//Build Classifiers
      ArrayList<String> names=new ArrayList<>();
     Classifier[] c = HandandBoneOutlines.setSingleClassifiers(names);    
     for(int i=0;i<c.length;i++){
        System.out.print(" CLASSIFIER ="+names.get(i));
        double accDistal=ClassifierTools.singleTrainTestSplitAccuracy(c[i], shapeletTrain,shapeletTest);
        System.out.println(" Acc ="+accDistal);
     }            
        
    }
    
    public static void summaryStatsTestTrain(){
        Instances train=ClassifierTools.loadData(path+"worms_TRAIN");
        Instances test=ClassifierTools.loadData(path+"worms_TEST");
        NormalizeCase nc = new NormalizeCase();
        try {
            train=nc.process(train);
            test=nc.process(test);
            SummaryStats ss=new SummaryStats();
            train=ss.process(train);
            test=ss.process(test);
//Build Classifiers
            ArrayList<String> names=new ArrayList<>();
            Classifier[] c = HandandBoneOutlines.setSingleClassifiers(names);    
            for(int i=0;i<c.length;i++){
               System.out.print(" CLASSIFIER ="+names.get(i));
               double accDistal=ClassifierTools.singleTrainTestSplitAccuracy(c[i], train,test);
               System.out.println(" Acc ="+accDistal);
            }            
        } catch (Exception ex) {
            Logger.getLogger(Worms.class.getName()).log(Level.SEVERE, null, ex);
        }
            
    }
    
    
    public static void main(String[] args){
        
int a,b=3,c=20,d=5;   double x=3.5;

System.out.println("a="+(b*c/d));
System.out.println("a="+b/c*d);
a=b/(c*d);
a=(b/c)*d;
a=c/b;
a=c%b;
//a=c/x;
System.out.println("a="+c/(int)x);
System.out.println("a="+(int)(c/x));

String test="switch";
switch(test){
    case "HELLO":
        System.out.println("Hello");
        break;
    
}

//        formatArff();        
 //        makeTestTrainSplit(); 
//       Instances data = ClassifierTools.loadData(path+"worms");
 //       basicShapeletTestTrain();
//        summaryStatsTestTrain();
    }
}
