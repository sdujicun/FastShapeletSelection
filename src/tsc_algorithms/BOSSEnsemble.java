package tsc_algorithms;

import fileIO.OutFile;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream; 
import java.io.IOException; 
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveCVAccuracy;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;

/**
 * BOSS classifier with parameter search and ensembling, if parameters are known, use 'BOSS' classifier and directly provide them.
 * 
 * Params: normalise? (i.e should first fourier coefficient(mean value) be discarded)
 * Alphabetsize fixed to four
 * 
 * @author James Large
 */
public class BOSSEnsemble implements Classifier, SaveCVAccuracy {
    
    private List<BOSSWindow> classifiers; 

    private final double correctThreshold = 0.92;
    private final Integer[] wordLengths = { 16, 14, 12, 10, 8 };
    private final int alphabetSize = 4;
    //private boolean norm;
    
    public enum SerialiseOptions { 
        //dont do any seriealising, run as normal
        NONE, 
        
        //serialise the final boss classifiers which made it into ensemble (does not serialise the entire BOSSEnsemble object)
        //slight runtime cost 
        STORE, 
        
        //serialise the final boss classifiers, and delete from main memory. reload each from ser file when needed in classification. 
        //the most memory used at any one time is therefore ~2 individual boss classifiers during training. 
        //massive runtime cost, order of magnitude 
        STORE_LOAD 
    };
    
    
    private SerialiseOptions serOption = SerialiseOptions.NONE;
    private static String serFileLoc = "BOSSWindowSers\\";
     
    private boolean[] normOptions;
    
    private String trainCVPath;
    private boolean trainCV=false;
    
    /**
     * Providing a particular value for normalisation will force that option, if 
     * whether to normalise should be a parameter to be searched, use default constructor
     * 
     * @param normalise whether or not to normalise by dropping the first Fourier coefficient
     */
    public BOSSEnsemble(boolean normalise) {
        normOptions = new boolean[] { normalise };
    }
    
    /**
     * During buildClassifier(...), will search through normalisation as well as 
     * window size and word length if no particular normalisation option is provided
     */
    public BOSSEnsemble() {
        normOptions = new boolean[] { true, false };
    }  

    public static class BOSSWindow implements Comparable<BOSSWindow>, Serializable { 
        private BOSS classifier;
        public double accuracy;
        public String filename;
        
        private static final long serialVersionUID = 2L;

        public BOSSWindow(String filename) {
            this.filename = filename;
        }
        
        public BOSSWindow(BOSS classifer, double accuracy, String dataset) {
            this.classifier = classifer;
            this.accuracy = accuracy;
            buildFileName(dataset);
        }

        public double classifyInstance(Instance inst) throws Exception { 
            return classifier.classifyInstance(inst); 
        }
        
        public double classifyInstance(int test) throws Exception { 
            return classifier.classifyInstance(test); 
        }
        
        private void buildFileName(String dataset) {
            filename = serFileLoc + dataset + "_" + classifier.windowSize + "_" + classifier.wordLength + "_" + classifier.alphabetSize + "_" + classifier.norm + ".ser";
        }
        
        public boolean storeAndClearClassifier() {
            try {
                ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
                out.writeObject(this);
                out.close();   
                clearClassifier();
                return true;
            }catch(IOException e) {
                System.out.print("Error serialiszing to " + filename);
                e.printStackTrace();
                return false;
            }
        }
        
        public boolean store() {
            try {
                ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
                out.writeObject(this);
                out.close();         
                return true;
            }catch(IOException e) {
                System.out.print("Error serialiszing to " + filename);
                e.printStackTrace();
                return false;
            }
        }
        
        public void clearClassifier() {
            classifier = null;
        }
        
        public boolean load() {
            BOSSWindow bw = null;
            try {
                ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename));
                bw = (BOSSWindow) in.readObject();
                in.close();
                this.accuracy = bw.accuracy;
                this.classifier = bw.classifier;
                return true;
            }catch(IOException i) {
                System.out.print("Error deserialiszing from " + filename);
                i.printStackTrace();
                return false;
            }catch(ClassNotFoundException c) {
                System.out.println("BOSSWindow class not found");
                c.printStackTrace();
                return false;
            }
        }
        
        public boolean deleteSerFile() {
            try {
                File f = new File(filename);
                return f.delete();
            } catch(SecurityException s) {
                System.out.println("Unable to delete, access denied: " + filename);
                s.printStackTrace();
                return false;
            }
        }
        
        /**
         * @return { numIntervals(word length), alphabetSize, slidingWindowSize } 
         */
        public int[] getParameters() { return classifier.getParameters();  }
        public int getWindowSize()   { return classifier.getWindowSize();  }
        public int getWordLength()   { return classifier.getWordLength();  }
        public int getAlphabetSize() { return classifier.getAlphabetSize(); }
        public boolean isNorm()      { return classifier.isNorm(); }
        
        @Override
        public int compareTo(BOSSWindow other) {
            if (this.accuracy > other.accuracy) 
                return 1;
            if (this.accuracy == other.accuracy) 
                return 0;
            return -1;
        }
    }
    
    @Override
    public void setCVPath(String train) {
        trainCVPath=train;
        trainCV=true;
    }

    @Override
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        
        BOSSWindow first = classifiers.get(0);
        sb.append("windowSize=").append(first.getWindowSize()).append("/wordLength=").append(first.getWordLength());
        sb.append("/alphabetSize=").append(first.getAlphabetSize()).append("/norm=").append(first.isNorm());
            
        for (int i = 1; i < classifiers.size(); ++i) {
            BOSSWindow boss = classifiers.get(i);
            sb.append(",windowSize=").append(boss.getWindowSize()).append("/wordLength=").append(boss.getWordLength());
            sb.append("/alphabetSize=").append(boss.getAlphabetSize()).append("/norm=").append(boss.isNorm());
        }
        
        return sb.toString();
    }
    
    @Override
    public int setNumberOfFolds(Instances data){
        return data.numInstances();
    }
    
     /**
     * @return { numIntervals(word length), alphabetSize, slidingWindowSize } for each BOSSWindow in this *built* classifier
     */
    public int[][] getParametersValues() {
        int[][] params = new int[classifiers.size()][];
        int i = 0;
        for (BOSSWindow boss : classifiers) 
            params[i++] = boss.getParameters();
         
        return params;
    }
    
    public void setSerOption(SerialiseOptions option) { 
        serOption = option;
    }
    
    public void setSerFileLoc(String path) {
        serFileLoc = path;
    }
    
    @Override
    public void buildClassifier(final Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSSEnsemble_BuildClassifier: Class attribute not set as last attribute in dataset");
 
        if (serOption == SerialiseOptions.STORE || serOption == SerialiseOptions.STORE_LOAD) {
            DateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
            Date date = new Date();
            serFileLoc += data.relationName() + "_" + dateFormat.format(date) + "\\";
            File f = new File(serFileLoc);
            if (!f.isDirectory())
                f.mkdirs();
        }
        
        classifiers = new LinkedList<BOSSWindow>();
        
        
        int numSeries = data.numInstances();
        
        int seriesLength = data.numAttributes()-1; //minus class attribute
        int minWindow = 10;
        int maxWindow = seriesLength; 

        //int winInc = 1; //check every window size in range
        
//        //whats the max number of window sizes that should be searched through
        //double maxWindowSearches = Math.min(200, Math.sqrt(seriesLength)); 
        double maxWindowSearches = seriesLength/4.0;
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches); 
        if (winInc < 1) winInc = 1;
        
        
        //keep track of current max window size accuracy, constantly check for correctthreshold to discard to save space
        double maxAcc = -1.0;
        
        for (boolean normalise : normOptions) {
            for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {          
                BOSS boss = new BOSS(wordLengths[0], alphabetSize, winSize, normalise);  
                boss.buildClassifier(data); //initial setup for this windowsize, with max word length     

                BOSS bestClassifierForWinSize = null; 
                double bestAccForWinSize = -1.0;

                //find best word length for this window size
                for (Integer wordLen : wordLengths) {            
                    boss = boss.buildShortenedBags(wordLen); //in first iteration, same lengths (wordLengths[0]), will do nothing

                    int correct = 0; 
                    for (int i = 0; i < numSeries; ++i) {
                        double c = boss.classifyInstance(i); //classify series i, while ignoring its corresponding histogram i
                        if (c == data.get(i).classValue())
                            ++correct;
                    }

                    double acc = (double)correct/(double)numSeries;     
                    if (acc >= bestAccForWinSize) {
                        bestAccForWinSize = acc;
                        bestClassifierForWinSize = boss;
                    }
                }

                //if not within correct threshold of the current max, dont bother storing at all
                if (bestAccForWinSize >= maxAcc * correctThreshold) {
                    BOSSWindow bw = new BOSSWindow(bestClassifierForWinSize, bestAccForWinSize, data.relationName());
                    bw.classifier.clean();
                    
                    if (serOption == SerialiseOptions.STORE)
                        bw.store();
                    else if (serOption == SerialiseOptions.STORE_LOAD)
                        bw.storeAndClearClassifier();
                        
                    classifiers.add(bw);
                    
                    if (bestAccForWinSize > maxAcc) {
                        maxAcc = bestAccForWinSize;       
                        //get rid of any extras that dont fall within the final max threshold
                        Iterator<BOSSWindow> it = classifiers.iterator();
                        while (it.hasNext()) {
                            BOSSWindow b = it.next();
                            if (b.accuracy < maxAcc * correctThreshold) {
                                if (serOption == SerialiseOptions.STORE || serOption == SerialiseOptions.STORE_LOAD)
                                    b.deleteSerFile();
                                it.remove();
                            }
                        }
                    }
                }
            }
        }
        
        if (trainCV) {
            int folds=setNumberOfFolds(data);
            OutFile of=new OutFile(trainCVPath);
            of.writeLine(data.relationName()+",BOSSEnsemble,train");
           
            double[][] results = findEnsembleTrainAcc(data);
            of.writeLine(getParameters());
            of.writeLine(results[0][0]+"");
            for(int i=1;i<results[0].length;i++)
                of.writeLine(results[0][i]+","+results[1][i]);
            System.out.println("CV acc ="+results[0][0]);
        }
    }

    private double[][] findEnsembleTrainAcc(Instances data) throws Exception {
        
        double[][] results = new double[2][data.numInstances() + 1];
        
        double correct = 0; 
        for (int i = 0; i < data.numInstances(); ++i) {
            double c = classifyInstance(i, data.numClasses()); //classify series i, while ignoring its corresponding histogram i
            if (c == data.get(i).classValue())
                ++correct;
            
            results[0][i+1] = data.get(i).classValue();
            results[1][i+1] = c;
        }
        
        results[0][0] = correct / data.numInstances();
        //TODO fill results[1][0]
        
        return results;
    }
    
    /**
     * Classify the train instance at index 'test', whilst ignoring the corresponding bags 
     * in each of the members of the ensemble, for use in CV of BOSSEnsemble
     */
    public double classifyInstance(int test, int numclasses) throws Exception {
        double[] dist = distributionForInstance(test, numclasses);
        
        double maxFreq=dist[0], maxClass=0;
        for (int i = 1; i < dist.length; ++i) 
            if (dist[i] > maxFreq) {
                maxFreq = dist[i];
                maxClass = i;
            }
        
        return maxClass;
    }

    public double[] distributionForInstance(int test, int numclasses) throws Exception {
        double[] classHist = new double[numclasses];
        
        //get votes from all windows 
        double sum = 0;
        for (BOSSWindow classifier : classifiers) {
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.load();
            double classification = classifier.classifyInstance(test);
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.clearClassifier();
            classHist[(int)classification]++;
            sum++;
        }
        
        if (sum != 0)
            for (int i = 0; i < classHist.length; ++i)
                classHist[i] /= sum;
        
        return classHist;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        
        double maxFreq=dist[0], maxClass=0;
        for (int i = 1; i < dist.length; ++i) 
            if (dist[i] > maxFreq) {
                maxFreq = dist[i];
                maxClass = i;
            }
        
        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] classHist = new double[instance.numClasses()];
        
        //get votes from all windows 
        double sum = 0;
        for (BOSSWindow classifier : classifiers) {
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.load();
            double classification = classifier.classifyInstance(instance);
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.clearClassifier();
            classHist[(int)classification]++;
            sum++;
        }
        
        if (sum != 0)
            for (int i = 0; i < classHist.length; ++i)
                classHist[i] /= sum;
        
        return classHist;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void main(String[] args) throws Exception{
//        basicTest();
//        ensembleMemberTest();
        resampleTest();
    }
    
    public static void basicTest() {
        System.out.println("BOSSEnsembleBasicTest\n");
        try {
            
            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TEST.arff");
//            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TRAIN.arff");
//            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TEST.arff");
            
            System.out.println(train.relationName());
            
            BOSSEnsemble boss = new BOSSEnsemble(true);
            System.out.println("Training starting");
            long start = System.nanoTime();
            boss.buildClassifier(train);
            double trainTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Training done (" + trainTime + "s)");
            
            System.out.println("Ensemble Size: " + boss.classifiers.size());
            System.out.println("Param sets: ");
            int[][] params = boss.getParametersValues();
            for (int i = 0; i < params.length; ++i)
                System.out.println(i + ": " + params[i][0] + " " + params[i][1] + " " + params[i][2] + " " + boss.classifiers.get(i).isNorm());
            
            System.out.println("\nTesting starting");
            start = System.nanoTime();
            double acc = ClassifierTools.accuracy(test, boss);
            double testTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Testing done (" + testTime + "s)");
            
            System.out.println("\nACC: " + acc);
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }
    
        public static void ensembleMemberTest() {
        System.out.println("BOSSEnsembleEnsembleMemberTest\n");
        try {
            
//            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN.arff");
//            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST.arff");
//            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TRAIN.arff");
//            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TEST.arff");
            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TEST.arff");
            
            System.out.println(train.relationName());
            
            BOSSEnsemble boss = new BOSSEnsemble(true);
            System.out.println("Training starting");
            long start = System.nanoTime();
            boss.buildClassifier(train);
            double trainTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Training done (" + trainTime + "s)");
            
            System.out.println("Ensemble Size: " + boss.classifiers.size());
            System.out.println("Param sets: ");
            int[][] params = boss.getParametersValues();
            for (int i = 0; i < params.length; ++i)
                System.out.println(i + ": " + params[i][0] + " " + params[i][1] + " " + params[i][2] + " " + boss.classifiers.get(i).isNorm());
            
            System.out.println("\nTesting starting");
            start = System.nanoTime();
            double ensembleacc = ClassifierTools.accuracy(test, boss);
            double testTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Testing done (" + testTime + "s)");
            
            System.out.println("\nACC: " + ensembleacc);
            
            
            for (BOSSWindow window : boss.classifiers) {
                BOSS individual = new BOSS(window.getWordLength(), window.getAlphabetSize(), window.getWindowSize(), true);
                
                System.out.println("\nTesting individual " + individual.getWordLength() + " " + individual.getAlphabetSize() + " " + individual.getWindowSize() + " " + individual.isNorm());
                individual.buildClassifier(train);

                if (!individual.bags.equals(window.classifier.bags)) {
                    System.out.println("**DIFFERENT BAGS**");
                }

                double acc = ClassifierTools.accuracy(test, individual);
                System.out.println("Individual classification acc: " + acc);
                System.out.println("Ensemble window *TRAINING* acc: " + acc);
            }
            
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }
        
        
     public static void resampleTest() throws Exception {
//         Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TRAIN.arff");
//         Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TEST.arff");
         Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TRAIN.arff");
         Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TEST.arff");
         
         BOSSEnsemble c = new BOSSEnsemble(true);
         
         //c.setSerOption(SerialiseOptions.STORE);
         //c.setSerOption(SerialiseOptions.STORE_LOAD);
         
         //c.setCVPath("C:\\tempproject\\BOSSEnsembleCVtest.csv");
         
         int resamples = 10;
         double [] accs = new double[resamples];
         
         for(int i=0;i<resamples;i++){
            Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);
            double act,pred;
            
                        
            c.buildClassifier(data[0]);
            accs[i]=0;
            for(int j=0;j<data[1].numInstances();j++)
            {
                act=data[1].instance(j).classValue();
                pred=c.classifyInstance(data[1].instance(j));
                if(act==pred)
                    accs[i]++;
                
            }
            accs[i]/=data[1].numInstances();
            System.out.println(accs[i]);
         }
         
         double mean = 0;
         for(int i=0;i<resamples;i++) {
             mean += accs[i];
         }
         mean/=resamples;
         System.out.println("Mean acc over " + resamples + " resamples: " + mean);
     }   
}