/*
Combines various files into standard results format.

Levels of collation at problem, classifier and split

1. single file by problem, classifier and split to single file of problem and classifier 
NaiveBayes/Adiac1.csv ....NaiveBayes/Adiac1.csv

to 
NaiveBayes/Adiac.csv
Adiac,0.6,0.53,0.55...

2. single file of problem and classifier  to single file of classifier
NaiveBayes/Adiac.csv
NaiveBayes/ArrowHead.csv

to 
NaiveBayes/NaiveBayes.csv
Where collated data is in format
problem, mean, std dev, sample size


3. single file for each classifier into single file
Results/NaiveBayes/NaiveBayes.csv
Results/NaiveBayes/C45.csv
to
Results/Accuracy.csv
Results/StDev.csv
Results/SampleSize.csv



*/
package bakeOffExperiments;

import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author ajb
 */
public class CollateResults {
   static final String[][] names={Experiments.standard,Experiments.elastic,Experiments.shapelet,Experiments.dictionary,Experiments.interval,Experiments.ensemble,Experiments.complexity};
   static String[] dirNames=Experiments.directoryNames;
   static int[] testSizes={391,175,30,20,20,60,900,3840,1380,28,250,390,390,390,306,276,139,139,139,100,4500,861,7711,1690,88,2050,455,175,1320,810,150,105,370,308,64,550,1980,1029,375,61,73,2345,60,760,291,154,154,1252,1965,1965,30,242,858,1896,105,291,205,205,375,375,180,600,375,601,953,8236,370,625,995,300,228,130,100,1139,4000,3582,3582,3582,3582,6164,54,638,77,77,3000,};
    static String[] c={"ST","ACF","PACF"};
    HashSet<String> finished=new HashSet<>();
/** 
 * 2. single file of problem and classifier  to single file of classifier
 * NaiveBayes/Adiac.csv
 * NaiveBayes/ArrowHead.csv    
 * */
    public static void generateAllScripts(String path, String classifier){
       boolean oldCls=false;
        int mem=8000;
        int maxMem=mem+2000;
        String queue,java; 
        String jar="LS.jar";
        if(oldCls){
            queue="long";
            java= "java/jdk/1.8.0_31";
        }
        else{
            queue="long-eth";
            java="java/jdk1.8.0_51";
        }
        File f=new File(path+"/"+classifier);
        deleteDirectory(f);
        if(!f.isDirectory())
            f.mkdir();
        for(int i=0;i<DataSets.fiveSplits.length;i++){
            OutFile of2;
            if(oldCls)
                of2=new OutFile(path+"/"+classifier+(i+1)+"OldCls.txt");
            else
                of2=new OutFile(path+"/"+classifier+(i+1)+".txt");
            for(int j=0;j<DataSets.fiveSplits[i].length;j++){
                String prob=DataSets.fiveSplits[i][j];
                OutFile of;
                if(oldCls)
                    of=new OutFile(path+"/"+classifier+"/"+prob+"OldCls.bsub");
                else
                    of=new OutFile(path+"/"+classifier+"/"+prob+".bsub");
                of.writeString("#!/bin/csh\n" +
                "#BSUB -q ");
                of.writeString(queue+"\n#BSUB -J ");
                of.writeLine(classifier+prob+"[1-100]");
                of.writeString("#BSUB -oo output/"+classifier+prob+"%I.out\n" +
                    "#BSUB -eo error/"+classifier+prob+"%I.err\n" +
                    "#BSUB -R \"rusage[mem="+mem+"]\"\n" +
                    "#BSUB -M "+maxMem);
                of.writeLine("\n\n module add "+java);

                of.writeLine("java -jar "+jar+" "+classifier+" "+prob+" $LSB_JOBINDEX ");
                if(oldCls)
                    of2.writeLine("bsub < "+"Scripts/"+classifier+"/"+prob+"OldCls.bsub");                
                else
                    of2.writeLine("bsub < "+"Scripts/"+classifier+"/"+prob+".bsub");
                of.closeFile();
            }
        }
    }
    public static void clusterResultsCollation(String path){
        for(int i=0;i<dirNames.length;i++){
            for(int j=0;j<names[i].length;j++){
//Check for directory
                File dir= new File(path+"\\"+dirNames[i]+"\\"+names[i][j]);
                if(dir.isDirectory()){    //Proceed if there is a directory of results
                    OutFile results = new OutFile(path+"\\"+dirNames[i]+"\\"+names[i][j]+".csv");
                    for(String s:DataSets.fileNames){
                        File f= new File(path+"\\"+dirNames[i]+"\\"+names[i][j]+"\\"+s+".csv");
                        if(f.exists()){
                            InFile f2=new InFile(path+"\\"+dirNames[i]+"\\"+names[i][j]+"\\"+s+".csv");
// Limit to a max of 100 here                          
                            String str=f2.readLine();
                            String[] spl=str.split(",");
                            int k;
//                            System.out.println("SPL length ="+spl.length);
                            for(k=0;k<spl.length && k<100;k++)
                                results.writeString(spl[k]+",");
                            if(k<spl.length&&k==100)
                                results.writeString(spl[k]+"\n");
                            else
                                results.writeString("\n");
                                
                        }
                    }
                    results.closeFile();
                }
 //               else{
   //                System.out.println(path+"\\"+dirNames[i]+"\\"+names[i][j]+" IS NOT a directory");
//
  //             }
            }
        }
    }
 
/* Takes a possibly partial list of results and format into the outfile */
    public static class Results{
        String name;
        double mean;
        double median;
        double stdDev;
        double[] accs;
        public boolean equals(Object o){
            if( ((Results)o).name.equals(this.name))
                return true;
            return false;
        }
    }
//OUTPUT TO C:\Users\ajb\Dropbox\Big TSC Bake Off\New Results\SingleClassifiers    
    public static void incompleteResultsParser(InFile f, OutFile of) throws Exception{
        int lines= f.countLines();
        f.reopen();
        ArrayList<Results> res=new ArrayList<>();
  //      System.out.println("Lines = "+lines);
        
        for(int i=0;i<lines;i++){
            String line=f.readLine();
            String[] split=line.split(",");
            Results r=new Results();
            r.mean=0;
            r.name=split[0];
            if(split.length>1){
                r.accs=new double[split.length-1];
    //            System.out.println("length="+r.accs.length+"::::"+line);
                for(int j=0;j<r.accs.length;j++){
                    try{
                        r.accs[j]=Double.parseDouble(split[j+1]);
//                        if(r.accs[j]>1)
//                           r.accs[j]=r.accs[j-1]; //REMOVE IMMEDIATELY
//                            throw new Exception("ERRPR ACCURACY >1 = "+r.accs[j]+" line ="+i+" split ="+j+" file = "+f.getName());
                        r.mean+=r.accs[j];
                    }catch(Exception e){
                        System.out.println("ERROR: "+split[j]+" giving error "+e+" in file "+f.getName()+" on line "+i+" name ="+r.name);
                        System.exit(0);
                    }
                }
                r.mean/=r.accs.length;
                r.stdDev=0;
                for(int j=0;j<r.accs.length;j++){
                    r.stdDev+=(r.accs[j]-r.mean)*(r.accs[j]-r.mean);
                }
                r.stdDev/=(r.accs.length-1);
                r.stdDev=Math.sqrt(r.stdDev);
                Arrays.sort(r.accs);
                if(r.accs.length%2==0)
                    r.median=(r.accs[r.accs.length/2]+r.accs[r.accs.length/2-1])/2;
                else
                    r.median=r.accs[r.accs.length/2];
                    
            }
            res.add(r);
        }
        for(int i=0;i<DataSets.fileNames.length;i++){
            of.writeString(DataSets.fileNames[i]+",");
            int j=0; //Wasteful linear scan
            while(j<res.size() && !DataSets.fileNames[i].equals(res.get(j).name))
                j++;
//            System.out.println("J =: "+j+" "+res.size());
            if(j<res.size()){
                Results r=res.get(j);
                if(r.mean>0)
                    of.writeLine(r.mean+","+r.stdDev+","+r.accs.length+","+r.median);
                else
                    of.writeLine("");
            }
            else
                of.writeLine("");
        }
    }    
    

    public static void fileStandardiseForProblems(String path) throws Exception{

        
        for(int i=0;i<dirNames.length;i++){
            for(String s:names[i]){
                File f= new File(path+"\\"+dirNames[i]+"\\"+s+".csv");
                if(f.exists()){
                    InFile inf=new InFile(path+"\\"+dirNames[i]+"\\"+s+".csv");
                    OutFile outf=new OutFile(path+"\\SingleClassifiers\\"+s+".csv");
                    incompleteResultsParser(inf,outf);
                    inf.closeFile();
                    outf.closeFile();
                }
                else
                    System.out.println(" File "+path+"\\"+dirNames[i]+"\\"+s+" does not exist");
            }
        }
    }
    public static void fileCombineClassifiers(String inPath,String outPath) throws Exception{
        OutFile[] of={new OutFile(outPath+"\\Means.csv"),new OutFile(outPath+"\\StDevs.csv"),new OutFile(outPath+"\\SampleSizes.csv"),new OutFile(outPath+"\\Medians.csv")};

        
        for(OutFile o:of){
            o.writeString(",");
            for(String[] n:names)
            for(String st:n)
                o.writeString(st+",");
        }
        for(OutFile o:of)
            o.writeString("\n");
        
//Try open all
        InFile[][] inFiles=new InFile[names.length][];
        for(int i=0;i<inFiles.length;i++){
            inFiles[i]=new InFile[names[i].length];
            for(int j=0;j<inFiles[i].length;j++){
//Check existence
                File f=new File(inPath+names[i][j]+".csv");
//If exists, open                
                if(f.exists())
                    inFiles[i][j]=new InFile(inPath+names[i][j]+".csv");
                else{
                    inFiles[i][j]=null;
//                    System.out.println(" File "+names[i][j]+" does not exist");
                }
            }
        }
        for(String s:DataSets.fileNames){
            for(OutFile o:of)
                o.writeString(s+",");
            for(int i=0;i<inFiles.length;i++){
                for(int j=0;j<inFiles[i].length;j++){
                    if(inFiles[i][j]==null){
                            of[0].writeString(",");
                            of[1].writeString(",");
                            of[2].writeString("0,");
                            of[3].writeString(",");
                    }
                    else{
                        String[] name=inFiles[i][j].readLine().split(",");
                        if(name.length<2){
                            of[0].writeString(",");
                            of[1].writeString(",");
                            of[2].writeString("0,");
                            of[3].writeString(",");
                        }
                        else{
                            for(int k=0;k<of.length;k++)
                                of[k].writeString(name[k+1]+",");
                        }
                    }
                }
            }    
            for(OutFile o:of)
               o.writeString("\n");
        }
    }
/*    
    public static void parseSingleProblem(String path,String result, String problem){
//Check they all exist        
        for(int i=0;i<100;i++){
            File f= new File(path+"\\fold"+i+".csv");
            InFile inf= new InFile(path+"\\fold"+i+".csv");
            int cases=inf.countLines();
            if(!f.exists() || cases==0){
                System.out.println(" Incomplete files, no fold "+i+" on path ="+path);
                System.exit(0);
            }
            inf.closeFile();
        }
        OutFile out=new OutFile(result);
        out.writeString(problem+",");
        for(int i=0;i<100;i++){
            InFile inf= new InFile(path+"\\fold"+i+".csv");
            int cases=inf.countLines();
            inf= new InFile(path+"\\fold"+i+".csv");
            double acc=0;
            for(int j=0;j<cases;j++){
                int act=inf.readInt();
                int pred=inf.readInt();
                if(act==pred){
                    acc++;
                }
            }
            acc/=cases;
            out.writeString(acc+",");
            System.out.println("Fold "+i+" acc ="+acc);
            inf.closeFile();
        }
    }
 */
    
    public static void combineFolds(String source, OutFile dest, int start, int end){
        for(int i=start;i<=end;i++){
            File inf=new File(source+"fold"+i+".csv");
//            System.out.println(" Reading "+inf.getPath());
            if(inf.exists() && inf.length()>0){
                InFile f=new InFile(source+"fold"+i+".csv");
                int lines=f.countLines();
                int testCount=0;
                if(lines>testCount){//SHOULD CHECK HERE Error, fold not complete
                    f=new InFile(source+"fold"+i+".csv");
                    double acc=0;
                    for(int j=0;j<lines;j++){
                        double act=f.readDouble();
                        double pred=f.readDouble();
                        if(act==pred) acc++;
                    }
                    if(i<end)
                        dest.writeString(acc/lines+",");
                    else
                        dest.writeLine(acc/lines+"");
                }else{
                    System.out.println("Error, "+inf.getPath()+" not complete only "+lines+" cases instead of "+testSizes[i]);
                }
            }
        }
    }

    public static void combineSingles(String root){
        OutFile out = new OutFile(root+"IncompleteFolds.csv");
        for(int i=0;i<dirNames.length;i++){
            for(int j=0;j<names[i].length;j++){
//Check for directory of
                File dir= new File(root+"\\"+dirNames[i]+"\\"+names[i][j]);
                if(dir.isDirectory()){    //Proceed if there is a directory of results
                    for(int k=0;k<DataSets.fileNames.length;k++){
                        String s= DataSets.fileNames[k];
                        dir= new File(root+"\\"+dirNames[i]+"\\"+names[i][j]+"\\Predictions"+"\\"+s);
    //Check if there is a directory of predictions.
                      if(dir.isDirectory()){
//                            int temp=checkPredictionLength(k,dir.getPath(),out);
//                            numFolds+=temp;
 //                           if(temp==100)
 //                               completeCount++;
                    //The files dir+"\\"+s+".csv" contain the average accuracy per fold. 
                            String p=root+"\\"+dirNames[i]+"\\"+names[i][j];
                            File f=new File(p+"\\"+s+".csv");
                            if(f.exists()){ //At least partially complete
        //See how complete it is
//                                System.out.println(" Reading "+p+"\\"+s+".csv");
                                InFile inf=new InFile(p+"\\"+s+".csv");
                                String line=inf.readLine();                            
                                inf.closeFile();
                                String[] res=null;
                                if(line==null){ //Delete the file
                                    f=new File(p+"\\"+s+".csv");
                                    f.delete();
                                }
                                else{
                                    res=line.split(",");
    //                                System.out.println(" line ="+line+" length = "+res.length);

                                }
                                if(res!=null && res.length<101)
                                {   //Check to see if there are any preds need adding
                                    OutFile of = new OutFile(p+"\\"+s+".csv");
                                    int length=res.length-1;
                                    for(String str:res){
                                        if(str.equals("NaN"))
                                            length--;
                                        else
                                            of.writeString(str+",");
                                    }
    //                                System.out.println("Checking for folds "+length+" to 99");
                                    combineFolds(p+"\\Predictions\\"+s+"\\",of,length,99);
                                    of.closeFile();

                                }
                            }
                            else{
                                //Check if there are any predictions
                                if(checkPredictions(p+"\\Predictions\\"+s+"\\")){
                                    OutFile of=new OutFile(p+"\\"+s+".csv"); 
                                    of.writeString(s+",");
                                    combineFolds(p+"\\Predictions\\"+s+"\\",of,0,99);
                                    of.closeFile();
                                }
                            }
                        }
                    }
                }
//                System.out.println("num folds for "+names[i][j]+" = "+numFolds+" complete data ="+completeCount);
            }
    /*        prob="MiddlePhalanxOutlineCorrect";
        of=new OutFile(root+"\\"+dir+"\\"+cls+"\\"+prob+"2.csv"); 
        combineFolds(root+"\\"+dir+"\\"+cls+"\\Predictions\\"+prob+"\\",of,27,99);
*/
        }
    }
    public static int checkPredictionLength(int problem, String path, OutFile of){
        System.out.println("Checking "+path);
        int count=0;
        boolean complete=true;
        for(int i=0;i<100;i++){
            File f2=new File(path+"\\fold"+i+".csv");
            if(f2.exists()){
                InFile f=new InFile(path+"\\fold"+i+".csv");
                int lines=f.countLines();
                if(lines!=testSizes[problem]){
                    of.writeLine(path+","+i+","+lines+","+testSizes[problem]);
//                    System.out.println("INCOMPLETE FOLD :"+path+","+i+","+lines+","+testSizes[problem]);
                    if(lines==0){
                        f.closeFile();
                        f2.delete();
                    }
                }
                else
                    count++;
            }
        }
        return count;
    }
    public static boolean checkPredictions(String path){
        for(int i=0;i<100;i++){
            if(new File(path+"fold"+i+".csv").exists())
                return true;
        }
        return false;
    }
    public static void generateParameterSplitScripts(String root, String dest,String classifier,String problem,int paras,int folds){
        InFile inf=new InFile(root+"\\SampleSizes.csv");
        File f=new File(dest+"\\Scripts\\"+problem);
        String jar="TimeSeriesClassification.jar";
        deleteDirectory(f);
        if(!f.isDirectory())
            f.mkdir();
        OutFile of2=new OutFile(dest+"\\Scripts\\"+problem+"\\"+problem+"paras.txt");
        for(int j=1;j<=paras;j++){
            OutFile of=new OutFile(dest+"\\Scripts\\"+problem+"\\paraFold"+"_"+j+".bsub");
            of.writeString("#!/bin/csh\n" +
                "#BSUB -q ");
            of.writeString("long\n#BSUB -J ");
            of.writeLine(classifier+problem+"[1-10]");
            of.writeString("#BSUB -oo output/"+classifier+"%I.out\n" +
              "#BSUB -eo error/"+classifier+"%I.err\n" +
              "#BSUB -R \"rusage[mem=2000]\"\n" +
              "#BSUB -M 3000");
            of.writeLine("\n\n module add java/jdk/1.8.0_31");
            of.writeLine("java -jar "+jar+" "+classifier+" " +problem+" $LSB_JOBINDEX"+j);
            of2.writeLine("bsub < "+"Scripts/"+classifier+"/Unstarted/"+problem+".bsub");
            of.closeFile();
        }
        
    }
    public static int getParas(String algo){
        switch(algo){
            case "TSBF": return 4;
            case "LS": return 8;
            default: return 10;    
        }
    }
    public static void generateScripts(String root, String dest){
        InFile inf=new InFile(root+"\\SampleSizes.csv");
        OutFile outf=new OutFile(dest+"\\AllProblems.txt");
        OutFile outf2=new OutFile(dest+"\\UnstartedProblems.txt");
        File f=new File(dest+"\\Scripts");
        boolean oldCls=false;
        int mem=9000;
        int maxMem=mem+1000;
        int maxNum=100;
        String jar="TimeSeriesClassification.jar";
        String queue,java; 
        if(oldCls){
            queue="short";
            java= "java/jdk/1.8.0_31";
        }
        else{
            queue="long-eth";
            java="java/jdk1.8.0_51";
        }
        deleteDirectory(f);
        if(!f.isDirectory())
            f.mkdir();
        String[] algos=new String[Experiments.numClassifiers()];
        for(int i=0;i<algos.length;i++)
           algos[i]=inf.readString();
        String[] problems=new String[85];
        int[][] counts=new int[85][algos.length];
        int c=0;
        int p=0;
        for(int i=0;i<problems.length;i++){
            problems[i]=inf.readString();
            for(int j=0;j<algos.length;j++){
                   counts[i][j]=inf.readInt();
//                System.out.print(counts[i][j]+" ");
            }
        }
        for(int j=0;j<algos.length;j++){
            if(generateScripts(algos[j])){
                for(int i=0;i<problems.length;i++){
                    if(counts[i][j]==0){
                        f=new File(dest+"\\Scripts\\"+algos[j]);
                        if(!f.isDirectory())
                            f.mkdir();
                        f=new File(dest+"\\Scripts\\"+algos[j]+"\\Unstarted");
                        if(!f.isDirectory())
                            f.mkdir();
                        c++;
                        p+=100;
                        int paras=getParas(algos[j]);
                        for(int k=1;k<=paras;k++){
                            OutFile of=new OutFile(dest+"\\Scripts\\"+algos[j]+"\\Unstarted\\"+problems[i]+"_"+k+".bsub");
                             of.writeString("#!/bin/csh\n" +
                                "#BSUB -q ");
                             of.writeString(queue+"\n#BSUB -J ");
                            of.writeLine(algos[j]+problems[i]+"["+(counts[i][j]+1)+"-"+maxNum+"]");
                            of.writeString("#BSUB -oo output/"+algos[j]+k+"%I.out\n" +
                                "#BSUB -eo error/"+algos[j]+k+"%I.err\n" +
                                "#BSUB -R \"rusage[mem="+mem+"]\"\n" +
                                "#BSUB -M "+maxMem);
                            of.writeLine("\n\n module add "+java);

                            of.writeLine("java -jar "+jar+" "+algos[j]+" " +problems[i]+" $LSB_JOBINDEX "+k);
                            outf2.writeLine("bsub < "+"Scripts/"+algos[j]+"/Unstarted/"+problems[i]+"_"+k+".bsub");
                            of.closeFile();
                        }
                    }
                    else if(counts[i][j]<100){                    
                        f=new File(dest+"\\Scripts\\"+algos[j]);
                        if(!f.isDirectory())
                            f.mkdir();
                        c++;
                        p+=(100-counts[i][j]);
                        if(algos[j].equals("TSBF")){
                            OutFile runScript=new OutFile(dest+"\\Scripts\\"+algos[j]+"\\Unfinished"+problems[i]+"Script.txt");
                            int paras=getParas(algos[j]);
                            for(int k=1;k<=paras;k++){
                                runScript.writeLine("bsub < Scripts/"+algos[j]+"/"+problems[i]+"_"+k+".bsub");
                                OutFile of=new OutFile(dest+"\\Scripts\\"+algos[j]+"\\"+problems[i]+"_"+k+".bsub");
                                 of.writeString("#!/bin/csh\n" +
                                  "#BSUB -q ");
                                 of.writeString(queue+"\n#BSUB -J ");
                                 if(counts[i][j]<=10)
                                    of.writeLine(algos[j]+problems[i]+"["+(counts[i][j]+1)+"-"+maxNum+"]");
                                 else
                                    of.writeLine(algos[j]+problems[i]+"["+(counts[i][j]+1)+"-100]");
                                of.writeString("#BSUB -oo output/"+algos[j]+k+"%I.out\n" +
                              "#BSUB -eo error/"+algos[j]+k+"%I.err\n" +
                              "#BSUB -R \"rusage[mem="+mem+"]\"\n" +
                              "#BSUB -M "+maxMem);
                                of.writeLine("\n\n module add "+java);

                                of.writeLine("java -jar "+jar+" "+algos[j]+" " +problems[i]+" $LSB_JOBINDEX "+k);
                                outf2.writeLine("bsub < "+"Scripts/"+algos[j]+"/Unstarted/"+problems[i]+"_"+k+".bsub");
                                of.closeFile();
                            }
                            runScript.closeFile();
                        }
                        else{
                            OutFile of=new OutFile(dest+"\\Scripts\\"+algos[j]+"\\"+problems[i]+".bsub");
                            of.writeString("#!/bin/csh\n" +
                              "#BSUB -q ");
                            if(counts[i][j]>0 && counts[i][j]<9){
                                of.writeString(queue+"\n#BSUB -J ");
                                of.writeLine(algos[j]+problems[i]+"["+(counts[i][j]+1)+"-"+maxNum+"]");
                                of.writeString("#BSUB -oo output/"+algos[j]+"%I.out\n" +
                              "#BSUB -eo error/"+algos[j]+"%I.err\n" +
                              "#BSUB -R \"rusage[mem="+mem+"]\"\n" +
                              "#BSUB -M "+maxMem);
                            }else{
                                of.writeString(queue+"\n#BSUB -J ");                           
                                of.writeLine(algos[j]+problems[i]+"["+(counts[i][j]+1)+"-100]");
                                of.writeString("#BSUB -oo output/"+algos[j]+"%I.out\n" +
                                  "#BSUB -eo error/"+algos[j]+"%I.err\n" +
                                  "#BSUB -R \"rusage[mem="+mem+"]\"\n" +
                                  "#BSUB -M "+maxMem);
                            }                            
                            of.writeLine("\n\n module add "+java);
                            of.writeLine("java -jar "+jar+" "+algos[j]+" " +problems[i]+" $LSB_JOBINDEX");
                            outf.writeLine("bsub < "+"Scripts/"+algos[j]+"/"+problems[i]+".bsub");
                            of.closeFile();
                        }
                    }
                }
            }
        }
        System.out.println(" Total number of problems remaining ="+c);
        System.out.println(" Total number of runs remaining ="+p);
        outf.closeFile();
    }
    public static boolean generateScripts(String algo){
        switch(algo){
            case "TSF":case "TSBF": case "LS": 
                return true;
            case "Logistic": case "MLP": 
            case "FS": case "ACF": case "PS": case "DTD_C":
            default:
                return false;
        }
    }
    public static boolean deleteDirectory(File directory) {
        if(directory.exists()){
            File[] files = directory.listFiles();
            if(null!=files){
                for(int i=0; i<files.length; i++) {
                    if(files[i].isDirectory()) {
                        deleteDirectory(files[i]);
                    }
                    else {
                        files[i].delete();
                    }
                }
            }
        }
        return(directory.delete());
    }    
    public static void main(String[] args){
      DataSets.resultsPath="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\";
//    generateAllScripts("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Scripts","LS");
        collateFoldZero();

      System.exit(0);
//        findNumberPerSplit();
        String root="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results";
//        generateScripts(root,root);        
//        System.exit(0);
        System.out.println("Combine singles ....");
        combineSingles(root);
        System.out.println("cluster results collation ....");
        clusterResultsCollation(root);
       try {
        System.out.println("file standardise for problems ....");
           fileStandardiseForProblems(root);
       } catch (Exception ex) {
           System.out.println("Eorr in fileStandardiseForProblems");
           System.exit(0);
       }
       try {
        System.out.println("file combine classifiers ....");
           fileCombineClassifiers(root+"\\SingleClassifiers\\",root);
       } catch (Exception ex) {
           System.out.println("Error in fileCombineClassifiers");
           System.exit(0);
       }
//        generateScripts(root,root);

    }
    public static void findNumberPerSplit(){
        String path="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\ensemble\\PS\\Predictions\\";
        int min;
            System.out.print("{");
        for(int i=0;i<DataSets.fileNames.length;i++){
            min=Integer.MAX_VALUE;
            int max=0;
            for(int j=0;j<100;j++){
                InFile f= new InFile(path+DataSets.fileNames[i]+"\\fold"+j+".csv");
                int t=f.countLines();
                if(t<min)
                    min=t;
                if(t>max)
                    max=t;
            }
            System.out.print(max+",");
        }
        System.out.print("};");
    }
    public static void collateFoldZero(){
        double[][] trainTestAcc=new double[DataSets.fileNames.length][Experiments.numClassifiers()];
        String[] allclassifiers=Experiments.allClassifiers();
        for(int i=0;i<trainTestAcc.length;i++){
            String prob=DataSets.fileNames[i];
            int pos=0;
//            for(int j=0;j<trainTestAcc[i].length;j++){
            for(int j=0;j<Experiments.classifiers.length;j++){
                for(int m=0;m<Experiments.classifiers[j].length;m++){
                    String cls=Experiments.classifiers[j][m];
        // Check to see if predictions 0 is present
                    String foldFile=DataSets.resultsPath+Experiments.directoryNames[j]+"\\"+cls+"\\Predictions\\"+prob+"\\fold0.csv";
//                    System.out.println("Looking for "+foldFile);
                    File f=new File(foldFile);
                    int size=0;
                    if(f.exists()){
            // if yes, check all the predictions are there
                        InFile inf=new InFile(foldFile);
                        size=inf.countLines();
                        inf.closeFile();
                    }
                    if(size==testSizes[i]){  // Complete fold
//                        System.out.println("fold 0 found ");
                        trainTestAcc[i][pos]=0;
                        InFile inf=new InFile(foldFile);
                        for(int k=0;k<size;k++){
                            String[] line=inf.readLine().split(",");
                            if(line[0].equals(line[1]))
                               trainTestAcc[i][pos]++;
                        }
                        trainTestAcc[i][pos]/=size;
                    }else{  //Try to recover from the single problem file

                        File f2=new File(DataSets.resultsPath+Experiments.directoryNames[j]+cls+"/"+prob+".csv");
  //                      System.out.println("\t\t fold 0 NOT found looking for "+f2.getPath());
                        if(f2.exists()){
                            InFile inf2=new InFile(f2.getPath());
                            String name=inf2.readString();
                            if(!name.equals(prob)){ //Error in name match
                                System.out.println("ERROR ALLIGNMENT FOR "+cls+" "+prob);
                                System.exit(0);
                            }
                            trainTestAcc[i][pos]=inf2.readDouble();
                        }
                        else{   //Try to recover from full file! 
 
                            File f3=new File(DataSets.resultsPath+Experiments.directoryNames[j]+"\\"+cls+".csv");
//                            System.out.println("\t\t"+f2.getName()+" NOT found looking for "+f3.getPath());
                            if(f3.exists()){
//                            System.out.println("\t\t"+f3.getName()+" FOUND");
                                InFile inf3=new InFile(f3.getPath());
                                int lines=inf3.countLines();
                                inf3=new InFile(f3.getPath());
                                boolean found=false;
                                for(int p=0;p<lines && !found;p++){
                                    String temp=inf3.readLine();
                                    if(temp!=null){
                                        String[] names=temp.split(",");
                                        if(names[0].equals(prob) && names.length>1){
                                            trainTestAcc[i][pos]=Double.parseDouble(names[1]);
                                            found=true;
                                        }
                                        else{
                                            if(names==null){
                                                trainTestAcc[i][pos]=-1;
                                                found=true;
                                                System.out.println("Error, "+cls+" "+prob+" not present in file"+f3.getName());
                                            }
                                        }
                                    }else{
                                        trainTestAcc[i][pos]=-1;
                                        found=true;
                                        System.out.println("Error, "+cls+" "+prob+" not present in file"+f3.getName());
                                    }
                                }
                            }
                            else{
                                System.out.println("Error, "+cls+" "+prob+" not present at all"); 
                                trainTestAcc[i][pos]=-1;
                            }
                         }

                    }
                        
                    pos++;
                }
            }
        }
        OutFile outf=new OutFile(DataSets.resultsPath+"singleTrainTest.csv");
        for(int i=0;i<allclassifiers.length;i++)
            outf.writeString(","+allclassifiers[i]);
        outf.writeString("\n");
        for(int i=0;i<trainTestAcc.length;i++){
            outf.writeString(DataSets.fileNames[i]);
            for(int j=0;j<trainTestAcc[i].length;j++){
                if(trainTestAcc[i][j]<=0)
                    outf.writeString(",");
                else
                    outf.writeString(","+trainTestAcc[i][j]);
            }
            outf.writeString("\n");          
        }
//First        
    }


}
