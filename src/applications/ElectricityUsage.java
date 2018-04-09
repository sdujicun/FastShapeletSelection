package applications;

import development.*;
import fileIO.OutFile;
import weka.core.Instances;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class ElectricityUsage {

   
    public static void fixData(){
        Instances test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+"ElectricDevices"+"\\"+"ElectricDevices"+"_TEST");
	Instances train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+"ElectricDevices"+"\\"+"ElectricDevices"+"_TRAIN");			
        int i=0;
        int c=0;
        while(i<train.numInstances()){
            int j=0;
            while(j<train.numAttributes()-1 && train.instance(i).value(j)==train.instance(i).value(0)){
                j++;
            }
            if(j==train.numAttributes()-1){    //Delete if constant
                train.delete(i);
                c++;
            }
            else
                i++;
        }
        System.out.println(" Removed "+c+" from train");
        c=0;
        i=0;
        while(i<test.numInstances()){
            int j=0;
            while(j<test.numAttributes()-1 && test.instance(i).value(j)==test.instance(i).value(0)){
                j++;
            }
            if(j==test.numAttributes()-1){    //Delete if constant
                test.delete(i);
                c++;
            }
            else
                i++;
        }
        System.out.println(" Removed "+c+" from test");
        OutFile of = new OutFile(DataSets.dropboxPath+"ElectricDevices"+"\\"+"ElectricDevices"+"_TEST.arff");
        OutFile of2 = new OutFile(DataSets.dropboxPath+"ElectricDevices"+"\\"+"ElectricDevices"+"_TRAIN.arff");
        of.writeLine(test.toString());
        of2.writeLine(train.toString());
        
//Test Filter
        try{
            NormalizeCase norm=new NormalizeCase();
            norm.setNormType(NormalizeCase.NormType.STD_NORMAL);
            test=norm.process(test);
//            train=norm.process(train);
        }catch(Exception e){
    //			System.out.println(trainSmall.toSting());
                            System.out.println(" Error with file+ \n"+test.instance(81));
                            double sum=0;
                            double sumSq=0;                            
                            for(int j=0;j<test.numAttributes()-1;j++){
                                sum+=test.instance(81).value(j);
                                sumSq+=test.instance(81).value(j)*test.instance(81).value(j);
                            }
                            System.out.println(" Sum ="+sum+" Sum Sq ="+sumSq+" num attribures ="+(test.numAttributes()-1));
                            
                            
                            e.printStackTrace();
                            System.exit(0);
            
        }
    }
 public static void main(String[] args){
    fixData();
 }
    
    
}
