package applications;

import fileIO.InFile;
import fileIO.OutFile;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.core.Instances;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.Differences;
import weka.filters.timeseries.RunLength;

public class BelkinChallenge {
    static String path="C:/Users/ajb/Dropbox/Belkin Competition/RealPower/";
    static int[] trainingDaysPerHouse={6,4,3,2};
    static String[][] fileNames={
        {
            "Tagged_Training_04_13_1334300401.mat",
            "Tagged_Training_10_22_1350889201.mat",
            "Tagged_Training_10_23_1350975601.mat",
            "Tagged_Training_10_24_1351062001.mat",
            "Tagged_Training_10_25_1351148401.mat",
            "Tagged_Training_12_27_1356595201.mat"
        },
        {
            "Tagged_Training_06_13_1339570801.mat",
            "Tagged_Training_06_14_1339657201.mat",
            "Tagged_Training_06_15_1339743601.mat",
            "Tagged_Training_02_15_1360915201.mat"
        },
        {
            "Tagged_Training_07_30_1343631601.mat",
            "Tagged_Training_07_31_1343718001.mat",
            "Tagged_Training_08_01_1343804401.mat"
            
        },
        {
            "Tagged_Training_07_26_1343286001.mat",
            "Tagged_Training_07_27_1343372401.mat"
        }  
    };
    
    public enum DeviceGroup{Light,ShortAppliance,LongAppliance,VariableAppliance};
    public static DeviceReading[][] allDevices;
    public static HouseholdReading[][] allHouseholds;

/* Activity detection needs to determine whether there is ANY device that turns on or off during any
 * given minute. First go:
 * 
 * Treat as separate problems. Each case is 360 readings. 
 * For data set 1, the positive case is any minute when a device is on. We may need to adjust the boundaries of the cases, 
 * because we want the on event at the centre of the positive case. 
 */
    public static void formatActivityOnOffProblem(boolean deviceOn,boolean r1){
        if(r1)
            allHouseholds=loadSingleSeriesData("Real1");
        else
            allHouseholds=loadSingleSeriesData("Real2");
        loadClassData();
        
        for(int i=0;i<4;i++){
            OutFile onOffDetection;
            if(deviceOn){
                if(r1)
                    onOffDetection=new OutFile(path+"BelkinR1OnDetectionHouse"+(i+1)+".arff");
                else
                    onOffDetection=new OutFile(path+"BelkinR2OnDetectionHouse"+(i+1)+".arff");
                onOffDetection.writeLine("@Relation BelkinOnDetection");
            }
            else{
               if(r1)
                    onOffDetection=new OutFile(path+"BelkinR1OffDetectionHouse"+(i+1)+".arff");
                else
                    onOffDetection=new OutFile(path+"BelkinR1OffDetectionHouse"+(i+1)+".arff");
                onOffDetection.writeLine("@Relation BelkinOnDetection");
            } 
            String channel="R2";
            if(r1)
                channel="R1";
                onOffDetection.writeLine("@Attribute UnixTimeStamp"+channel+" real");
            for(int j=0;j<360;j++)
                onOffDetection.writeLine("@Attribute RealPower"+channel+"_"+(j+1)+" real");
            onOffDetection.writeLine("@attribute deviceOn {0,1}");
            onOffDetection.writeLine("@data");
 //Extract out and concatinate a household.
//Find length            
            int totalLength=allHouseholds[i][0].time.length;
            for(int j=1;j<allHouseholds[i].length;j++)
                totalLength+=allHouseholds[i][j].time.length;
            System.out.println("Total Length for household "+(i+1)+" = "+totalLength);
//Form single series
            long[] unixTime=new long[totalLength];
            double[] watts=new double[totalLength];
            int count =0;
            for(int j=0;j<allHouseholds[i].length;j++){
                for(int k=0;k<allHouseholds[i][j].time.length;k++){
                    unixTime[count]=allHouseholds[i][j].unixSeconds[k];
                    watts[count]=allHouseholds[i][j].watts[k];
                    count++;
                }
            }
//For each device in the house 
        int start=0;    //Set start time to 30 seconds from the beginning
        int positiveCases=0;
        int negativeCases=0;
        for(DeviceReading d: allDevices[i]){
    //Form positive cases: For each device reading,         
            long eventPoint;
            if(deviceOn)
                eventPoint=d.start;
            else
               eventPoint=d.end;
            try{
                int index=Arrays.binarySearch(unixTime,eventPoint);
                onOffDetection.writeString(unixTime[index]+",");
                for(int j=0;j<360;j++)
                    onOffDetection.writeString(watts[index-180+j]+",");
                onOffDetection.writeString("1\n");
                positiveCases++;
            }catch(Exception e){
                System.out.println("Exception in find positive cases="+e);
                System.out.println(" Household ="+(i+1));
                System.out.println(" Device ="+d);
                System.out.println(" Looking for the index with a linear scan");
                for(int k=0;k<unixTime.length;k++)
                    if(unixTime[k]==eventPoint)
                        System.out.println("Found time "+eventPoint+" in position ="+k);
                
                System.exit(0);
                
            }
    //Form negative cases from minutes prior to the start of d. Since the original problem is manufactured, the 
            //class frequencies will not necessarily match the test data.
            //Set the start point for negative cases to 60 mins before the device is activated, as long
            //As that is in advance of the previous start point
           int index=Arrays.binarySearch(unixTime, eventPoint);
           if(unixTime[start]<unixTime[index]-60*60)
               start=index-30*60*6;
            try{
                while(unixTime[start+3*360]<eventPoint){
                    //Form a negative case 
                onOffDetection.writeString(unixTime[start]+",");
                    for(int j=0;j<360;j++)
                        onOffDetection.writeString(watts[start+j]+",");
                    onOffDetection.writeString("0\n");
//                System.out.println("Negative Case start = "+unixTime[start]+" End = "+unixTime[start+360]+" prior to device start ="+d.start);
                    start+=360;
                    negativeCases++;
                }
    //Look up the position of the end index            
                start=Arrays.binarySearch(unixTime, d.end);
    //Increment 30 minutes the end
                start+=30*360;
            }catch(Exception e){
                System.out.println("Exception in find negative cases="+e);
                System.out.println(" Household ="+(i+1));
                System.out.println(" Device ="+d);
                System.out.println("  index ="+eventPoint+" start unix time ="+unixTime[start]);
                System.exit(0);
                
            }
        }
        System.out.println("DETECTION HOUSE "+(i+1)+": Positive Cases= "+positiveCases+" Negative Cases = "+negativeCases);

//        OutFile offDetection=new OutFile(path+"BelkinOffDetection.csv");
        
        
        
    }
        
        
 }
    
 
    
    public static void formatDeviceDetectionProblem(){
        HouseholdReading[][] a1=loadSingleSeriesData("Real1");
        HouseholdReading[][] a2=loadSingleSeriesData("Real2");
        loadClassData();             
        for(int i=0;i<4;i++){
            OutFile of =new OutFile(path+"DeviceSeriesHouse"+(i+1)+".csv");
//Extract out and concatinate a household.
//Find length            
            int totalLength1=a1[i][0].time.length;
            for(int j=1;j<a1[i].length;j++)
                totalLength1+=a1[i][j].time.length;
            System.out.println("Total Length for household R1"+(i+1)+" = "+totalLength1);
//Form single series
            long[] unixTime1=new long[totalLength1];
            double[] watts1=new double[totalLength1];
            int count =0;
            for(int j=0;j<a1[i].length;j++){
                for(int k=0;k<a1[i][j].time.length;k++){
                    unixTime1[count]=a1[i][j].unixSeconds[k];
                    watts1[count]=a1[i][j].watts[k];
                    count++;
                }
            }
 //Find length            
            int totalLength2=a2[i][0].time.length;
            for(int j=1;j<a2[i].length;j++)
                totalLength2+=a2[i][j].time.length;
            System.out.println("Total Length for household R2"+(i+1)+" = "+totalLength2);
//Form single series
            long[] unixTime2=new long[totalLength2];
            double[] watts2=new double[totalLength2];
            count =0;
            for(int j=0;j<a2[i].length;j++){
                for(int k=0;k<a2[i][j].time.length;k++){
                    unixTime2[count]=a2[i][j].unixSeconds[k];
                    watts2[count]=a2[i][j].watts[k];
                    count++;
                }
            }                       
            for(DeviceReading d: allDevices[i]){
//Extract out the series for each device on R1 and R2 separately
                 int startIndex=Arrays.binarySearch(unixTime1, d.start);    
                 int endIndex=Arrays.binarySearch(unixTime1, d.end);
//take 25 secs either side'
                 int offset=25;
                 startIndex-=6*offset;
                 endIndex+=6*offset;
//Write R1 to file
                 of.writeString(unixTime1[startIndex]+",");
                 for(int j=startIndex;j<endIndex;j++)
                        of.writeString(watts1[j]+","); 
                 of.writeString(d.deviceName+"\n"); 
//Extract out the series for each device on R1 and R2 separately
                 startIndex=Arrays.binarySearch(unixTime2, d.start);    
                 endIndex=Arrays.binarySearch(unixTime2, d.end);
//take 15 secs either side
                 startIndex-=6*offset;
                 endIndex+=6*offset;
//Write R1 to file
                 of.writeString(unixTime2[startIndex]+",");
                 for(int j=startIndex;j<endIndex;j++)
                        of.writeString(watts2[j]+","); 
                 of.writeString(d.deviceName+"\n"); 
            }
        }
 }
    
 
    public static void formatDeviceClassificationProblem(){
        HouseholdReading[][] a1=loadSingleSeriesData("Real1");
        HouseholdReading[][] a2=loadSingleSeriesData("Real2");
        loadClassData();             
        for(int i=0;i<4;i++){
            OutFile of =new OutFile(path+"DeviceSeriesHouse"+(i+1)+".csv");
//Extract out and concatinate a household.
//Find length            
            int totalLength1=a1[i][0].time.length;
            for(int j=1;j<a1[i].length;j++)
                totalLength1+=a1[i][j].time.length;
            System.out.println("Total Length for household R1"+(i+1)+" = "+totalLength1);
//Form single series
            long[] unixTime1=new long[totalLength1];
            double[] watts1=new double[totalLength1];
            int count =0;
            for(int j=0;j<a1[i].length;j++){
                for(int k=0;k<a1[i][j].time.length;k++){
                    unixTime1[count]=a1[i][j].unixSeconds[k];
                    watts1[count]=a1[i][j].watts[k];
                    count++;
                }
            }
 //Find length            
            int totalLength2=a2[i][0].time.length;
            for(int j=1;j<a2[i].length;j++)
                totalLength2+=a2[i][j].time.length;
            System.out.println("Total Length for household R2"+(i+1)+" = "+totalLength2);
//Form single series
            long[] unixTime2=new long[totalLength2];
            double[] watts2=new double[totalLength2];
            count =0;
            for(int j=0;j<a2[i].length;j++){
                for(int k=0;k<a2[i][j].time.length;k++){
                    unixTime2[count]=a2[i][j].unixSeconds[k];
                    watts2[count]=a2[i][j].watts[k];
                    count++;
                }
            }                       
            for(DeviceReading d: allDevices[i]){
//Extract out the series for each device on R1 and R2 separately
                 int startIndex=Arrays.binarySearch(unixTime1, d.start);    
                 int endIndex=Arrays.binarySearch(unixTime1, d.end);
//take 25 secs before startt time, and set end time to three minutes after start
                 int offset=25;
                 int endOffset=180;
                 startIndex-=6*offset;
                 endIndex=startIndex+6*endOffset;
//Write R1 to file
                 of.writeString(unixTime1[startIndex]+",");
                 for(int j=startIndex;j<endIndex;j++)
                        of.writeString(watts1[j]+","); 
                 of.writeString(d.deviceName+"\n"); 
//Extract out the series for each device on R1 and R2 separately
                 startIndex=Arrays.binarySearch(unixTime2, d.start);    
                 endIndex=Arrays.binarySearch(unixTime2, d.end);
//take 15 secs either side
                 startIndex-=6*offset;
                 endIndex+=6*offset;
//Write R1 to file
                 of.writeString(unixTime2[startIndex]+",");
                 for(int j=startIndex;j<endIndex;j++)
                        of.writeString(watts2[j]+","); 
                 of.writeString(d.deviceName+"\n"); 
            }
        }
 }
    
 
        
    public static HouseholdReading[][] loadSingleSeriesData(String s){
        
        HouseholdReading[][] allH=new HouseholdReading[4][];
        for(int i=0;i<4;i++){
            allH[i]=new HouseholdReading[trainingDaysPerHouse[i]];
            for(int j=0;j<trainingDaysPerHouse[i];j++){
                InFile f = new InFile(path+"H"+(i+1)+"/"+fileNames[i][j]+s+".csv");
                int size=f.countLines();
                f = new InFile(path+"H"+(i+1)+"/"+fileNames[i][j]+s+".csv");
                allH[i][j]=new HouseholdReading(f,size);
            }
        }
        return allH;
    }
    public static void combineProblems(){
        for(int i=1;i<=4;i++){
            System.out.println("*********** HOUSE NUMBER ********** "+i);
            Instances data = ClassifierTools.loadData(path+"BelkinR1OnDetectionHouse"+i);
            Instances data2 = ClassifierTools.loadData(path+"BelkinR2OnDetectionHouse"+i);
            Instances data3;
            data.setClassIndex(-1);
            data.deleteAttributeAt(data.numAttributes()-1);
            data2.deleteAttributeAt(0);
            data3=Instances.mergeInstances(data, data2);
            OutFile of = new OutFile(path+"BelkinR1R2OnDetectionHouse"+i+".arff");
            of.writeString(data3.toString());
        }
    }
    
    public static void classificationTest(String channel) throws Exception{
        //On detection
        ArrayList<String> names=new ArrayList<>();
        Classifier[] c= ClassifierTools.setDefaultSingleClassifiers(names);
//On Classification
        String path="C:\\Users\\ajb\\Dropbox\\Belkin Competition\\RealPower\\";
        OutFile results=new OutFile(path+"DeviceOn"+channel+"Classification.csv");
        results.writeString(",");
        for(String s:names)
            results.writeString(s+",");
        results.writeString("\n");
        for(int i=1;i<=4;i++){
            System.out.println("*********** HOUSE NUMBER ********** "+i);
            Instances data = ClassifierTools.loadData(path+"Belkin"+channel+"OnDetectionHouse"+i);
//Delete the time index            
            data.deleteAttributeAt(0);
            OutFile diffFile=new OutFile(path+"DeviceOnDiffs"+i+".csv");
            NormalizeCase norm=new NormalizeCase();
            Instances normData=norm.process(data);
            Differences diff=new Differences();
            diff.setOrder(1);
            Instances diffData = diff.process(data);
            diffFile.writeLine(diffData+"\n");
            Instances normDiffData= diff.process(normData);
            results.writeString("House"+i+"RAW,");
            for(int j=0;j<c.length;j++){
                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(c[j],data, data.numInstances(), new Random(1));            
                System.out.println(" RAW Classifier "+names.get(j)+" accuracy ="+((double)eval.correct())/data.numInstances());
                results.writeString(((double)eval.correct())/data.numInstances()+",");
            }
/*             results.writeString("House"+i+"RawNorm,");
            for(int j=0;j<c.length;j++){
                Evaluation eval = new Evaluation(normData);
                eval.crossValidateModel(c[j],normData, normData.numInstances(), new Random(1));            
                System.out.println("RawNorm Classifier "+names.get(j)+" accuracy ="+((double)eval.correct())/data.numInstances());
                results.writeString(((double)eval.correct())/data.numInstances()+",");
            }          
  */          results.writeString("House"+i+"FirstDifferences,");
            for(int j=0;j<c.length;j++){
                Evaluation eval = new Evaluation(diffData);
                eval.crossValidateModel(c[j],diffData, diffData.numInstances(), new Random(1));            
                System.out.println("Diff Classifier "+names.get(j)+" accuracy ="+((double)eval.correct())/data.numInstances());
                results.writeString(((double)eval.correct())/data.numInstances()+",");
            }
            results.writeString("House"+i+"NormFirstDifferences,");
            for(int j=0;j<c.length;j++){
                Evaluation eval = new Evaluation(normDiffData);
                eval.crossValidateModel(c[j],normDiffData, normDiffData.numInstances(), new Random(1));            
                System.out.println("Norm Diff Classifier "+names.get(j)+" accuracy ="+((double)eval.correct())/data.numInstances());
                results.writeString(((double)eval.correct())/data.numInstances()+",");
            }
            
        }
        
    }
    
    public static void loadClassData(){
        allDevices=new DeviceReading[4][];
        for(int i=1;i<=4;i++){
            InFile f=new InFile(path+"H"+i+"/AllTaggingInfoH"+i+".csv");
            int devices=f.countLines()-1;
            allDevices[i-1]=new DeviceReading[devices];
            f=new InFile(path+"H"+i+"/AllTaggingInfoH"+i+".csv");
            f.readLine();
            for(int j=0;j<devices;j++){
                String line=f.readLine();
                try{
                    allDevices[i-1][j]=new DeviceReading(line);
                }catch(Exception e){
                    System.out.println("Exception "+e+" on House "+i+" line "+j+" String ="+line);
                    e.printStackTrace();
                    System.exit(0);
                }
                    
            }
        }
    }

    
    public static class HouseholdReading{
        double[] time;
        long[] unixSeconds; 
        double[] watts;
        public HouseholdReading(InFile f, int size){
            time=new double[size];
            watts=new double[size];
            unixSeconds=new long[size]; 
            for(int i=0;i<size;i++){
                time[i]=f.readDouble();
                watts[i]=f.readDouble();
                unixSeconds[i]=(long)time[i];
            }
        }
    }
    
    public static class DeviceReading{
        String deviceName;
        int deviceCode;
        long start;
        long end;
        DeviceGroup type;
        public DeviceReading(String line){
            String[] split=line.split(",");
            deviceCode=Integer.parseInt(split[0]);
            deviceName=split[1];
            start=Long.parseLong(split[2]);
            end=Long.parseLong(split[3]);
        }
        public String toString(){
            return deviceName+","+start+","+end;
        }
    }
     public static void checkDifferences() throws Exception{
 
        String path="C:\\Users\\ajb\\Dropbox\\Belkin Competition\\RealPower\\";
        for(int i=1;i<=4;i++){
            System.out.println("*********** HOUSE NUMBER ********** "+i);
            Instances data = ClassifierTools.loadData(path+"BelkinOnDetectionHouse"+i);
//Delete the time index            
            data.deleteAttributeAt(0);
            OutFile diffFile=new OutFile(path+"DeviceOnDiffs"+i+".csv");
            OutFile diffFile2=new OutFile(path+"DeviceOnDiffs"+i+".csv");
            NormalizeCase norm=new NormalizeCase();
            Instances normData=norm.process(data);
            Differences diff=new Differences();
            diff.setOrder(1);
            Instances diffData = diff.process(data);
            diffFile.writeLine(diffData+"\n");
            Instances normDiffData= diff.process(normData);
            diffFile2.writeLine(normDiffData+"\n");
            
        }
        
    }
    
   
    public static void main(String[] args){
        formatDeviceDetectionProblem();
        System.exit(0);
        
/*combineProblems();        
      System.exit(0);
        formatActivityOnOffProblem(true,true);
        formatActivityOnOffProblem(true,false);
        formatActivityOnOffProblem(false,true);
        formatActivityOnOffProblem(false,true);
        System.exit(0);
 */       try{
   //     checkDifferences();
        classificationTest("R1");
        classificationTest("R2");
        classificationTest("R1R2");
        }catch(Exception e){
            System.out.println(" Exception ="+e);
            e.printStackTrace();
        }
        
//        loadClassData();
//        loadSingleSeriesData("Real1");
    }
}
