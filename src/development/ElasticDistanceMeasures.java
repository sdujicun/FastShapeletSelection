/*
Code to recreate 1-NN classifier results with alternative distance measures. 

*/
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.*;
import weka.core.*;
import weka.core.elastic_distance_measures.*;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class ElasticDistanceMeasures extends Thread{
    int threadNum;
    String[] names;
    OutFile results;
    private ElasticDistanceMeasures(){}
    public ElasticDistanceMeasures(int th, String[] s, OutFile of){
        threadNum=th;
        names=s;
        results=of;
    }

         @Override
    public void run() {
        try{
        testDTWCV();
        }catch(Exception e){
            results.closeFile();
            System.out.println(" Error un test DTWCV thread "+threadNum+" quitting");
            System.exit(0);
        }
    }

      public static double testErrorResampleExperiment(EuclideanDistance dc, String s,int seed, Classifier c) throws Exception{   //EuclideanDistance d
        Instances train = ClassifierTools.loadData(s+"_TRAIN");
        Instances test = ClassifierTools.loadData(s+"_TEST");
        Random r = new Random();
        r.setSeed(seed);
        NormalizeCase nc = new NormalizeCase();
        train=nc.process(train);
        test=nc.process(test);
        Instances all = new Instances(train);
        for(Instance in:test)
            all.add(in);
        all.randomize(r);
        int testSize=test.numInstances();
        train = new Instances(all);
        test= new Instances(all,0);
        for(int i=0;i<testSize;i++)
            test.add(train.remove(0));
       
        kNN knn= new kNN(1);
        knn.normalise(false);
        knn.setDistanceFunction(dc);
        double a1=ClassifierTools.singleTrainTestSplitAccuracy(knn, train, test);
        return (1-a1);
    }  
    
/** Run on the 46 UCR Data sets **/    
    public static void testUCR(EuclideanDistance dc) throws Exception{   //EuclideanDistance d
        OutFile of=new OutFile(DataSets.problemPath+"UCRResults"+dc.getClass().getName()+".csv");//d.getClass().getName()+
       for(String s:DataSets.ucrNames){
            Instances train = ClassifierTools.loadData(DataSets.problemPath+s+"\\"+s+"_TRAIN");
            Instances test = ClassifierTools.loadData(DataSets.problemPath+s+"\\"+s+"_TEST");
            NormalizeCase nc = new NormalizeCase();
            train=nc.process(train);
            test=nc.process(test);
            kNN knn= new kNN(1);
            knn.normalise(false);
            knn.setDistanceFunction(dc);
            double a1=ClassifierTools.singleTrainTestSplitAccuracy(knn, train, test);
            DecimalFormat df = new DecimalFormat("##.###");
            System.out.println(s+" train size"+train.numInstances()+" test size,"+test.numInstances()+"series length "+(test.numAttributes()-1)+", 1nn"+dc.getClass().getName()+","+df.format(1-a1));
            of.writeLine(s+","+(1-a1));
        }
    }
    static double distance(Instance a, Instance b){
        double[] d1=a.toDoubleArray();
        double[] d2=b.toDoubleArray();
        double dist=0;
        for(int i=0;i<d1.length-1;i++)
            dist+=(d1[i]-d2[i])*(d1[i]-d2[i]);
        return Math.sqrt(dist);
    }
    static int nearestNeighbour(Instances train, Instance test){
        double d;
        double minDist=Double.MAX_VALUE;
        int predClass=0;
        for(Instance tr:train){
            d=distance(tr,test);
            if(d<=minDist){
                minDist=d;
                predClass=(int)tr.classValue();
            }
        }
        return predClass;
    }
    static double nearestNeighbourAcc(Instances train, Instances test){
        double acc=0;
        int correct=0;
        for(Instance ins:test){
            int pred=nearestNeighbour(train,ins);
          // System.out.println(" pred ="+pred);
            if(pred==ins.classValue())
                correct++;
        }
        acc=correct/(double)test.numInstances();
        return acc;
    }
    public static void dtwTimingComparison(){
        OutFile of = new OutFile(DataSets.dropboxPath+"DTW_Comparison.csv");
        of.writeLine("Problem,DTWBasicA,DTWBasicT,DTWBasicEarlyA,DTWBasicEarlyT,DTW_DistanceBasicA,DTW_DistanceBasicT,DTW_DistanceEfficientA,DTW_DistanceEfficientT,DTW_DistanceSpaceEfficientA,DTW_DistanceSpaceEfficientT");
        
        for(String s:DataSets.ucrNames){
            System.out.print(s+",");
            of.writeString(s+",");
            EuclideanDistance[] dtw= new EuclideanDistance[4];
            Instances train = ClassifierTools.loadData(DataSets.dropboxPath+s+"\\"+s+"_TRAIN");
            Instances test = ClassifierTools.loadData(DataSets.dropboxPath+s+"\\"+s+"_TEST");
                  
            dtw[0] = new BasicDTW();
            dtw[1] =  new DTW_DistanceBasic();
            dtw[2] =  new DTW_DistanceEfficient();
//            dtw[3] =  new DTW_DistanceSpaceEfficient();
            for(int i=0;i<dtw.length;i++){
                kNN dist= new kNN(1);
                dist.setCrossValidate(false);
                dist.normalise(true);
                dist.setDistanceFunction(dtw[i]);
                long t1=System.nanoTime();
                double a = ClassifierTools.singleTrainTestSplitAccuracy(dist, train, test);
                long t2=System.nanoTime();
                double time=(t2-t1)/1000000.0;
                System.out.print(a+","+time+",");
                of.writeString(a+","+time+",");
            }            
            of.writeString("\n");
            System.out.print("\n");
            
        }
   
        
    }
    
    public void testBasicDTWCV() throws Exception{   //EuclideanDistance d
  //      OutFile of=new OutFile(DataSets.ucrPath+"UCRResultsDTWCVFull.csv");//d.getClass().getName()+
       for(String s:names){
            Instances train = ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TRAIN");
            Instances test = ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TEST");
/*            NormalizeCase nc = new NormalizeCase();
            train=nc.process(train);
            test=nc.process(test);
 */         
//Do CV here
/*            for(int i=1;i<=100;i++){
                EuclideanDistance d= new SakoeChibaDTW(i/100.0);
            KNN = new KNN(1);
            
            long t1=System.nanoTime();
            double a1=ClassifierTools.singleTrainTestSplitAccuracy(dtw, train, test);
            long t2=System.nanoTime();
            double time=(t2-t1)/10000000.0;
            DecimalFormat df = new DecimalFormat("##.###");
            System.out.println(s+":: 1nnDTWCV Test Acc="+df.format(1-a1)+" time="+df.format(time)+" best window="+dtw.getBestWarp());
            results.writeLine(s+","+(1-a1)+","+","+time);
 */       }
    }


    
    public void testDTWCV() throws Exception{   //EuclideanDistance d
  //      OutFile of=new OutFile(DataSets.ucrPath+"UCRResultsDTWCVFull.csv");//d.getClass().getName()+
       for(String s:names){
            Instances train = ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TRAIN");
            Instances test = ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TEST");
/*            NormalizeCase nc = new NormalizeCase();
            train=nc.process(train);
            test=nc.process(test);
*/
            DTW_1NN dtw=new DTW_1NN();
            dtw.optimiseWindow(true);
//            kNN dtw= new kNN(1);
//            dtw.setDistanceFunction(new BasicDTW());
            
            long t1=System.nanoTime();
            double a1=ClassifierTools.singleTrainTestSplitAccuracy(dtw, train, test);
            long t2=System.nanoTime();
            double time=(t2-t1)/10000000.0;
            DecimalFormat df = new DecimalFormat("##.#####");
            System.out.println(s+":: 1nnDTWCV Test Acc="+df.format(1-a1)+" time="+df.format(time));
//            System.out.println(s+":: 1nnDTWCV Test Acc="+df.format(1-a1)+" time="+df.format(time)+" best window="+dtw.getBestWarp()+" best R value ="+dtw.getR());
            results.writeLine(s+","+(1-a1)+","+","+time);
        }
    }
    public static void singleProblem(String s) throws Exception{   //EuclideanDistance d
            Instances train = ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TRAIN");
            Instances test = ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TEST");
/*            NormalizeCase nc = new NormalizeCase();
        train=nc.process(train);
        test=nc.process(test);
*/
        DTW_1NN dtw=new DTW_1NN();
        dtw.setMaxNosWindows(1);
        dtw.optimiseWindow(true);
        long t1=System.nanoTime();
        double a1=ClassifierTools.singleTrainTestSplitAccuracy(dtw, train, test);
        long t2=System.nanoTime();
        double time=(t2-t1)/10000000.0;
        DecimalFormat df = new DecimalFormat("##.#####");
        System.out.println(s+":: 1nnDTWCV Test Acc="+df.format(1-a1)+" time="+df.format(time)+" best window="+dtw.getBestWarp()+" best R value ="+dtw.getR());

    }

    
    public static void dtwEarlyAbandonTest(){
        OutFile of = new OutFile(DataSets.dropboxPath+"DTW_EarlyAbandon.csv");
        of.writeLine("Problem,DTW_DistanceBasicA,DTW_DistanceBasicT,DTW_DistanceEfficientA,DTW_DistanceEfficientT");
        
        for(String s:DataSets.ucrNames){
            System.out.print(s+",");
            of.writeString(s+",");
            EuclideanDistance[] dtw= new EuclideanDistance[2];
            Instances train = ClassifierTools.loadData(DataSets.dropboxPath+s+"\\"+s+"_TRAIN");
            Instances test = ClassifierTools.loadData(DataSets.dropboxPath+s+"\\"+s+"_TEST");
                  
            dtw[0] =  new DTW_DistanceBasic();
            dtw[1] =  new DTW_DistanceEfficient();
//            dtw[3] =  new DTW_DistanceSpaceEfficient();
            for(int i=0;i<dtw.length;i++){
                kNN dist= new kNN(1);
                dist.setCrossValidate(false);
                dist.normalise(true);
                dist.setDistanceFunction(dtw[i]);
                long t1=System.nanoTime();
                double a = ClassifierTools.singleTrainTestSplitAccuracy(dist, train, test);
                long t2=System.nanoTime();
                double time=(t2-t1)/1000000.0;
                System.out.print(a+","+time+",");
                of.writeString(a+","+time+",");
            }            
            of.writeString("\n");
            System.out.print("\n");
            
        }
   
        
    }
   
    public static String[][] getAllFiles(int threads){
        int size=DataSets.ucrNames.length/threads;
        int count=0;
        String[][] files= new String[threads][];
        for(int i=0;i<threads;i++){
            if(i!=threads-1){
                files[i]=new String[size];
                for(int j=0;j<files[i].length;j++)
                    files[i][j]=DataSets.ucrNames[count++];
            }
            else{
                files[i]=new String[DataSets.ucrNames.length-count];
                for(int j=0;j<files[i].length;j++)
                    files[i][j]=DataSets.ucrNames[count++];
            }
        }
        return files;
    }
    public static String[][] getSingleFiles(int threads){
        String[][] files= new String[threads][1];
        files[0][0]="CBF";
/*       files[1][0]="Cricket_Y";
       files[2][0]="Cricket_Z";
        files[3][0]="FacesUCR";
       files[4][0]="fiftywords";
       files[5][0]="fish";
       files[6][0]="Haptics";
       files[7][0]="CBF";
  */     return files;
    }
    
    public static void threadRun(String fileName) throws Exception{
        int threads=Runtime.getRuntime().availableProcessors();
        System.out.println(" nos processors ="+Runtime.getRuntime().availableProcessors());

        ElasticDistanceMeasures[] exp1=new ElasticDistanceMeasures[threads];
        OutFile[] of =new OutFile[threads];
  //    String[][] files=getSingleFiles(threads);
        
        String[][] files=getAllFiles(threads);
        for(int i=0;i<threads;i++){
            of[i]=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\DTW\\"+fileName+(i+1)+".csv");
            exp1[i]=new ElasticDistanceMeasures(i,files[i],of[i]);
            exp1[i].start();
        }
        for(int i=0;i<threads;i++)
            exp1[i].join();
    }   

        static class Results implements Comparable<Results>{
        String name;
        double acc;
        double time;
        
        public Results(InFile f){
            name=f.readString();
            acc=f.readDouble();
            time=f.readDouble();
        }

        @Override
        public int compareTo(Results o) {

            return name.toLowerCase().compareTo(o.name.toLowerCase());
            
        }
        public String toString(){
            return name+","+acc+","+time;
        }

    }
    
    public static void combineResults(int threads,String prob){
        String path="C:\\Users\\ajb\\Dropbox\\Results\\DTW\\";
        final OutFile results = new OutFile(path+"Combined"+prob+".csv");
        ArrayList<Results> all=new ArrayList<>();
        for(int i=1;i<=threads;i++){
            InFile in = new InFile(path+prob+i+".csv");
            int c=in.countLines();
            in = new InFile(path+prob+i+".csv");
            for(int j=0;j<c;j++)
                all.add(new Results(in));
        }
        
        Collections.sort(all);
       for(Results r:all)
        results.writeLine(r.toString());    
    }
    
    public static void CBFTest() throws Exception{   //EuclideanDistance d
        String s="CBF";
            Instances train = ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TRAIN");
            Instances test = ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TEST");
/*            NormalizeCase nc = new NormalizeCase();
        train=nc.process(train);
        test=nc.process(test);
*/      
        DTW_1NN dtw=new DTW_1NN();
        dtw.optimiseWindow(false);
        for(int i=0;i<=100;i++){
            dtw.setR(i/100.0);
            double a1=ClassifierTools.singleTrainTestSplitAccuracy(dtw, train, test);
            DecimalFormat df = new DecimalFormat("##.###");
            System.out.println(i+","+(i/100.0)+","+dtw.getWindowSize()+" 1nnDTWCV Test Acc="+df.format(1-a1));

        }
    }

    
    public static void main(String[] args) throws Exception{
       String prob="DTWCV_V3_";
//       singleProblem("CBF");
      combineResults(8,prob);
    System.exit(0);
//Cricket Z debug
 //   System.exit(0);
        threadRun(prob);        
       System.exit(0);
        OutFile of=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\DTW\\CricketX.csv");
        String[] str={"Cricket_X"};
        ElasticDistanceMeasures exp1=new ElasticDistanceMeasures(0,str,of);
        exp1.run();
        //exp1=new ElasticDistanceMeasures(1,str,of);
        //exp1.run();
        
        

//*/     
        
//        testDTWCV();
 //       testUCR(new EuclideanDistance());//new EuclideanDistance()
//            dtwTimingComparison();
//                dtwEarlyAbandonTest();         
        
           }

}
