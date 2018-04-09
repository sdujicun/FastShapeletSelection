/*
 */
package tsc_algorithms;

import development.DataSets;
import fileIO.OutFile;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.text.DecimalFormat;
import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.trees.REPTree;
import weka.core.Attribute;
import weka.core.ContingencyTables;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.Utils;

/**
 *
 * @author ajb. Implementation of the learned pattern similarity algorithm
 * by M. Baydogan
 * 
 */
public class LPS extends AbstractClassifier implements ParameterSplittable{
    RandomRegressionTree[] trees;
    
    public static final int PARASEARCH_NOS_TREES=25;
    public static final int DEFAULT_NOS_TREES=200;    
    int nosTrees=DEFAULT_NOS_TREES;
    int nosSegments=20;
    double[] ratioLevels={0.01,0.1,0.25,0.5};
    double[] segmentProps={0.05,0.1,0.25,0.5,0.75,0.95};
    double segmentProp=segmentProps[0];
    double ratioLevel=ratioLevels[0];
    int[] treeDepths={2,4,6};
    int treeDepth=treeDepths[2];
    int[] segLengths;
    int[][] segStarts;
    int[][] segDiffStarts;
    Instances sequences;
    int[] nosLeafNodes;
    int[][][] leafNodeCounts;
    double[] trainClassVals;
    int[] classAtt;
    boolean paramSearch=true;
    double acc=0;
    public LPS(){
        trees=new RandomRegressionTree[nosTrees];
    }

    public String globalInfo() {
        return "Blah";
    }
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;

        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "M. Baydogan and G. Runger");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        result.setValue(TechnicalInformation.Field.TITLE, "Time series representation and similarity based on local\n" +
    "autopatterns");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "Online First");
        result.setValue(TechnicalInformation.Field.NUMBER, "");
        result.setValue(TechnicalInformation.Field.PAGES, "");
        return result;
      }

 //<editor-fold defaultstate="collapsed" desc="problems used in DAMI paper">   
    public static String[] problems={
        "Adiac",
        "ArrowHead",
//        "ARSim",
        "Beef",
        "BeetleFly",
        "BirdChicken",
        "Car",
        "CBF",
        "ChlorineConcentration",
        "CinCECGtorso",
        "Coffee",
        "Computers",
        "CricketX",
        "CricketY",
        "CricketZ",
        "DiatomSizeReduction",
        "DistalPhalanxOutlineAgeGroup",
        "DistalPhalanxOutlineCorrect",
        "DistalPhalanxTW",
        "Earthquakes",
        "ECGFiveDays",
        "ElectricDevices",
        "FaceAll",
        "FaceFour",
        "FacesUCR",
        "Fiftywords",
        "Fish",
        "FordA",
        "FordB",
        "GunPoint",
        "Haptics",
        "Herring",
        "InlineSkate",
        "ItalyPowerDemand",
        "LargeKitchenAppliances",
        "Lightning2",
        "Lightning7",
        "Mallat",
        "MedicalImages",
        "MiddlePhalanxOutlineAgeGroup",
        "MiddlePhalanxOutlineCorrect",
        "MiddlePhalanxTW",
        "MoteStrain",
        "NonInvasiveFatalECGThorax1",
        "NonInvasiveFatalECGThorax2",
        "OliveOil",
        "OSULeaf",
        "PhalangesOutlinesCorrect",
        "Plane",
        "ProximalPhalanxOutlineAgeGroup",
        "ProximalPhalanxOutlineCorrect",
        "ProximalPhalanxTW",
        "RefrigerationDevices",
        "ScreenType",
        "ShapeletSim",
        "ShapesAll",
        "SmallKitchenAppliances",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "StarLightCurves",
        "SwedishLeaf",
        "Symbols",
        "SyntheticControl",
        "ToeSegmentation1",
        "ToeSegmentation2",
        "Trace",
        "TwoLeadECG",
        "TwoPatterns",
        "UWaveGestureLibraryX",
        "UWaveGestureLibraryY",
        "UWaveGestureLibraryZ",
        "UWaveGestureLibraryAll",
        "Wafer",
        "WordSynonyms",
        "Yoga"};
      //</editor-fold>  
    

//<editor-fold defaultstate="collapsed" desc="results reported in DAMI paper">        
    static double[] reportedResults={
        0.211,
        0.2,
//        0.004,
        0.367,
        0.15,
        0.05,
        0.183,
        0.002,
        0.352,
        0.064,
        0.071,
        0.136,
        0.282,
        0.208,
        0.305,
        0.049,
        0.237,
        0.234,
        0.327,
        0.331,
        0.155,
        0.273,
        0.242,
        0.04,
        0.098,
        0.213,
        0.094,
        0.09,
        0.223,
        0,
        0.562,
        0.398,
        0.494,
        0.053,
        0.157,
        0.197,
        0.411,
        0.093,
        0.297,
        0.523,
        0.208,
        0.497,
        0.114,
        0.183,
        0.147,
        0.133,
        0.134,
        0.226,
        0,
        0.112,
        0.172,
        0.278,
        0.329,
        0.44,
        0.006,
        0.218,
        0.225,
        0.225,
        0.123,
        0.033,
        0.072,
        0.03,
        0.027,
        0.077,
        0.1,
        0.02,
        0.061,
        0.014,
        0.189,
        0.263,
        0.253,
        0.025,
        0.001,
        0.27,
        0.136
    };
      //</editor-fold>  
    
    
    
 public static void compareToPublished() throws Exception{
     DecimalFormat df=new DecimalFormat("###.###");
     OutFile res=new OutFile(DataSets.path+"recreatedLPS.csv");
     int b=0;
     int t=0;
     System.out.println("problem,recreated,published");
     for(int i=0;i<problems.length;i++){
         String s=problems[i];
        System.out.print(s+",");
        Instances train = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TRAIN.arff");
        Instances test = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TEST.arff");
        LPS l=new LPS();
        l.setParamSearch(false);
        l.buildClassifier(train);
        double a=ClassifierTools.accuracy(test, l);
        System.out.println(df.format(1-a)+","+df.format(reportedResults[i])+","+df.format(1-a-reportedResults[i]));
        if((1-a)<reportedResults[i])
            b++;
        if((1-a)==reportedResults[i])
            t++;
        res.writeLine(s+","+(1-a)+","+reportedResults[i]);
     }
     System.out.println("Reported better ="+(problems.length-t-b)+" ties ="+t+" ours better = "+b);
 } 
    
    @Override
    public void setParamSearch(boolean b) {
        paramSearch=b;
    }

    @Override
    public void setPara(int x) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String getParas() {
        return ratioLevel+","+treeDepth;
    }

    @Override
    public double getAcc() {
        return acc;
    }
    
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
//determine minimum and maximum possible segment length
        if(paramSearch){
            double bestErr=1;
            int bestRatio=0;
            int bestTreeDepth=0;
            LPS trainer=new LPS();
            trainer.nosTrees=50;
            trainer.setParamSearch(false);
            int folds=10;
            for(int i=0;i<ratioLevels.length;i++){
                trainer.ratioLevel=ratioLevels[i];
                for(int j=0;j<treeDepths.length;j++){
                    trainer.treeDepth=treeDepths[j];
                    Evaluation eval=new Evaluation(data);
                    eval.crossValidateModel(trainer, data, folds,new Random());
                    double e=eval.errorRate();
                    if(e<bestErr){
                        bestErr=e;
                        bestTreeDepth=j;
                        bestRatio=i;
                    }
                }
            }
            ratioLevel=ratioLevels[bestRatio];
            treeDepth=treeDepths[bestTreeDepth];
            System.out.println("Best ratio level ="+ratioLevel+" best tree depth ="+treeDepth+" with CV error ="+bestErr);
        }
        
        
        int seriesLength=data.numAttributes()-1;
        int minSegment=(int)(seriesLength*0.1);
        int maxSegment=(int)(seriesLength*0.9);
        segLengths=new int[nosTrees];
        nosLeafNodes=new int[nosTrees];
        segStarts=new int[nosTrees][nosSegments];
        segDiffStarts=new int[nosTrees][nosSegments];
        leafNodeCounts=new int[data.numInstances()][nosTrees][];
        trainClassVals=new double[data.numInstances()];
        for(int i=0;i<data.numInstances();i++)
            trainClassVals[i]=data.instance(i).classValue();
        classAtt=new int[nosTrees];
        Random r= new Random();
        
//For each tree 1 to N
        for(int i=0;i<nosTrees;i++){    
//    %select random segment length for each tree
            segLengths[i]=minSegment+r.nextInt(maxSegment-minSegment);
//    %select target segments randomly for each tree
//   %ind=1:(2*nsegment);            
//            int target=r.nextInt(2*nosSegments);    //times 2 for diffs
//        %construct segment matrix (both observed and difference)
//        stx=randsample(tlen-segmentlen(i),nsegment,true); 
//        stxdiff=randsample(tlen-segmentlen(i)-1,nsegment,true);
//Sample with replacement.
            for(int j=0;j<nosSegments;j++){
                segStarts[i][j]=r.nextInt(seriesLength-segLengths[i]);
                segDiffStarts[i][j]=r.nextInt(seriesLength-segLengths[i]-1);
            }
//Set up the instances for this tree            
//2- Generate segments for each time series and 
//        concatenate these segments rowwise, let resulting matrix be M
            FastVector atts=new FastVector();
            String name;
            for(int j=0;j<2*nosSegments;j++){
                    name = "SegFeature"+j;
                    atts.addElement(new Attribute(name));
            }
            sequences = new Instances("SubsequenceIntervals",atts,segLengths[i]*data.numInstances());            
            
            for(int j=0;j<data.numInstances();j++){
                Instance series=data.instance(j);
                for(int k=0;k<segLengths[i];k++){
                    DenseInstance in=new DenseInstance(sequences.numAttributes());
                    for(int m=0;m<nosSegments;m++)
                        in.setValue(m, series.value(segStarts[i][m]+k));
                    for(int m=0;m<nosSegments;m++)
                        in.setValue(nosSegments+m, series.value(segDiffStarts[i][m]+k)-series.value(segDiffStarts[i][m]+k+1));                     
                    sequences.add(in);    
//                  System.out.println(" TRAIN INS ="+in+" CLASS ="+series.classValue());

                }
            }
//3- Choose a random target column from M, let this target column be t
            classAtt[i]=r.nextInt(sequences.numAttributes());//
            sequences.setClassIndex(classAtt[i]);
            trees[i]= new RandomRegressionTree();
            trees[i].setMaxDepth(treeDepth);
            trees[i].setKValue(1);
//            System.out.println("Min Num ="+(int)(sequences.numInstances()*ratioLevel));
            trees[i].setMinNum((int)(sequences.numInstances()*ratioLevel));//leafratio*size(segments,1)
            trees[i].buildClassifier(sequences);
            nosLeafNodes[i]=trees[i].nosLeafNodes;
//            System.out.println("Num of leaf nodes ="+trees[i].nosLeafNodes);
            for(int j=0;j<data.numInstances();j++){
                leafNodeCounts[j][i]=new int[trees[i].nosLeafNodes];
                for(int k=0;k<segLengths[i];k++){
                    trees[i].distributionForInstance(sequences.instance(j*segLengths[i]+k));
                    int leafID=RandomRegressionTree.lastNode;
//                    System.out.println("Seq Number ="+(j*segLengths[i]+k));
                    leafNodeCounts[j][i][leafID]++;
                }
            }
            
//Set up no pruning, minimum number at leaf nodes to leafratio*size(segments,1),
//nvartosample means only single variable considered at each node.             
//  splitting consider only one random column, namely r and find the split value.
//        tree = classregtree(segments(:,ind~=target(i)),segments(:,target(i)),'method','regression', ...
//            'prune','off','minleaf',leafratio*size(segments,1),'nvartosample',1);
                    
        }
//        System.out.println(" Nos Sequence Cases ="+sequences.numInstances());
/*        for (int i = 0; i < data.numInstances(); i++) {
//Find the leaf node of every subsequence belonging to instance i for every tree
            System.out.print("Instance "+i+" HIST: ");
            for(int j=0;j<leafNodeCounts[i].length;j++)
                for(int k=0;k<leafNodeCounts[i][j].length;k++)
                    System.out.print(leafNodeCounts[i][j][k]+" ");
            System.out.print(" CLASS ="+data.instance(i).classValue()+" \n ");
        }
  */      
        sequences=null;
        System.gc();
     }
    public double distance(int[][] test, int[][] train){
        double d=0;
        for(int i=0;i<test.length;i++)
            for(int j=0;j<test[i].length;j++){
                double x=(test[i][j]-train[i][j]);
                if(x>0)
                    d+=x;
                else
                    d+=-x;
            }
        return d;
    }
    public double classifyInstance(Instance ins) throws Exception{
        
        int[][] testNodeCounts=new int[nosTrees][];
//Extract sequences, shove them into instances. 
//        concatenate these segments rowwise, let resulting matrix be M
            

        for(int i=0;i<nosTrees;i++){    
            FastVector atts=new FastVector();
            String name;
            for(int j=0;j<2*nosSegments;j++){
                    name = "SegFeature"+j;
                    atts.addElement(new Attribute(name));
            }
            sequences = new Instances("SubsequenceIntervals",atts,segLengths[i]);            
            for(int k=0;k<segLengths[i];k++){
                DenseInstance in=new DenseInstance(sequences.numAttributes());
                for(int m=0;m<nosSegments;m++)
                    in.setValue(m, ins.value(segStarts[i][m]+k));
                for(int m=0;m<nosSegments;m++)
                    in.setValue(nosSegments+m, ins.value(segDiffStarts[i][m]+k)-ins.value(segDiffStarts[i][m]+k+1));
                sequences.add(in);
//                System.out.println(" TEST INS ="+in+" CLASS ="+ins.classValue());
            }            
            sequences.setClassIndex(classAtt[i]);
            testNodeCounts[i]=new int[trees[i].nosLeafNodes];
            for(int k=0;k<sequences.numInstances();k++){
                trees[i].distributionForInstance(sequences.instance(k));
                int leafID=RandomRegressionTree.lastNode;
//                    System.out.println("Seq Number ="+(j*segLengths[i]+k));
                testNodeCounts[i][leafID]++;
            }
        }
//        System.out.println(" TEST NODE COUNTS =");
//        for(int i=0;i<testNodeCounts.length;i++){
//            for(int j=0;j<testNodeCounts[i].length;j++)
//                System.out.print(" "+testNodeCounts[i][j]);
//            System.out.println("");
//        }
//        System.out.println(" TRAIN NODE COUNTS =");
//        for(int k=0;k<leafNodeCounts.length;k++){
//            for(int i=0;i<leafNodeCounts[k].length;i++){
//                for(int j=0;j<leafNodeCounts[k][i].length;j++)
//                    System.out.print(" "+leafNodeCounts[k][i][j]);
//                System.out.println("");
//            }
//        }
            
//1-NN on the counts
        double minDist=Double.MAX_VALUE;
        int closest=0;
        for(int i=0;i<leafNodeCounts.length;i++){
            double d=distance(testNodeCounts,leafNodeCounts[i]);
            if(d<minDist){
                minDist=d;
                closest=i;
            }
        }
        return trainClassVals[closest];
    }
        public static Object readFromFile(String filename) {  
            Object obj=null;
            try{
                FileInputStream fis = new FileInputStream(filename);
                ObjectInputStream in = new ObjectInputStream(fis);
                obj =in.readObject();
                in.close();
            }
            catch(Exception ex){
                ex.printStackTrace();
           }                      
            return obj;

        }

    public static void main(String[] args) throws Exception {
        
//       compareToPublished();
//        System.exit(0);
        LPS l=new LPS();
        l.setParamSearch(false);
        String prob="ItalyPowerDemand"; 
        double mean=0;
        Instances train = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+prob+"\\"+prob+"_TRAIN.arff");
        Instances test = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+prob+"\\"+prob+"_TEST.arff");
//        Instances train = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\Code\\Baydogan LPS\\Train.arff");
//        Instances test = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\Code\\Baydogan LPS\\Test.arff");
//        train.setClassIndex(train.numAttributes()-1);
//        test.setClassIndex(test.numAttributes()-1);
//        System.out.println("Train = "+train);
//        System.out.println("Test = "+test);
        l.buildClassifier(train);
        double a=ClassifierTools.accuracy(test, l);
        System.out.println( "test prob accuracy = "+a);
    }

    
/**
 * After obtaining the ensemble, what I do is to find out which rows of M goes 
 * in to what terminal node of each tree. 
 * 
 * Let's consider one tree. Rows of M extracted from time series S are residing 
 * 
 * in particular nodes of this tree. 
 * I characterize each time series by the number of rows residing in each 
 * terminal node. 
 * 
 * 
 * 
 * When I do the same for all trees in the ensemble, it is 
 * all about combining these terminal node distribution vectors into one 
 * long vector and compute similarity over this single vector. 
 * Without loss of generality, suppose I have 16 terminal nodes for each tree 
 * in my ensemble of 10 trees. That will result in a representation vector of 
 * length 16x10=160. Then I compute the similarity (actually dissimilarity) 
 * by taking the sum of absolute differences.

1,2,3,4,5,6,7,8
8,7,6,5,4,3,2,1
Let l=3, nsegs =2, start pos be 2 and 4
Series 1
Seg 1: 2,3,4
Seg 2: 4,5,6
Series 2
Seg 1: 7,6,5
Seg 2: 5,4,3

M equals
2,4
3,5 
4,6
7,5
6,4
5,3
    **/    
    public void debugFeatureExtraction(){
      //determine minimum and maximum possible segment length

            FastVector atts2=new FastVector();
            for(int j=0;j<9;j++){
                    atts2.addElement(new Attribute("SegFeature"+j));
            }
            double[] t1={1,2,3,4,5,6,7,8};
            double[] t2={8,7,6,5,4,3,2,1};
         Instances data= new Instances("SubsequenceIntervals",atts2,2);            
         DenseInstance ins=new DenseInstance(data.numAttributes());
         for (int i = 0; i < t1.length; i++) {
            ins.setValue(i, t1[i]);
        }
         data.add(ins);
         ins=new DenseInstance(data.numAttributes());
         for (int i = 0; i < t2.length; i++) {
            ins.setValue(i, t2[i]);
        }
         data.add(ins);
         System.out.println("TEST DATA ="+data);
         nosSegments=2;
         nosTrees=1;
        int seriesLength=data.numAttributes()-1;
        int minSegment=(int)(seriesLength*0.1);
        int maxSegment=(int)(seriesLength*0.9);
        segLengths=new int[nosTrees];
        segStarts=new int[nosTrees][nosSegments];
        segDiffStarts=new int[nosTrees][nosSegments];
        Random r= new Random();
        
//For each tree 1 to N
        for(int i=0;i<nosTrees;i++){    
//    %select random segment length for each tree
            segLengths[i]=minSegment+r.nextInt(maxSegment-minSegment);
            segLengths[i]=3;
            System.out.println("SEG LENGTH ="+segLengths[i]);
//    %select target segments randomly for each tree
//   %ind=1:(2*nsegment);            
            int target=r.nextInt(2*nosSegments);    //times 2 for diffs
//        %construct segment matrix (both observed and difference)
//        stx=randsample(tlen-segmentlen(i),nsegment,true); 
//        stxdiff=randsample(tlen-segmentlen(i)-1,nsegment,true);
//Sample with replacement.
            for(int j=0;j<nosSegments;j++){
                segStarts[i][j]=r.nextInt(seriesLength-segLengths[i]);
                segDiffStarts[i][j]=r.nextInt(seriesLength-segLengths[i]-1);
                System.out.println("SEG START ="+segStarts[i][j]);
                System.out.println("SEG DIFF START ="+segDiffStarts[i][j]);
            }
//Set up the instances for this tree            
            Instances tr=null;     
            FastVector atts=new FastVector();
            String name;
            for(int j=0;j<2*nosSegments;j++){
                    name = "SegFeature"+j;
                    atts.addElement(new Attribute(name));
            }
            Instances result = new Instances("SubsequenceIntervals",atts,segLengths[i]*data.numInstances());            
            
            for(int j=0;j<data.numInstances();j++){
                Instance series=data.instance(j);
                for(int k=0;k<segLengths[i];k++){
                    DenseInstance in=new DenseInstance(result.numAttributes());
                    for(int m=0;m<nosSegments;m++)
                        in.setValue(m, series.value(segStarts[i][m]+k));
                    for(int m=0;m<nosSegments;m++)
                        in.setValue(nosSegments+m, series.value(segDiffStarts[i][m]+k)-series.value(segDiffStarts[i][m]+k+1));                     
                    result.add(in);                    
                }
            }
            System.out.println("DESIRED OUTPUT : ");
            System.out.println("2,4\n" +
                "3,5\n" +
                "4,6\n" +
                "7,5\n" +
                "6,4\n" +
                "5,3");
            System.out.println("TRANSFORMED INSTANCES ="+result);
        }
  
    }

}
