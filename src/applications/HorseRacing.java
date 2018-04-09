/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package applications;

import utilities.ClassifierTools;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class HorseRacing {
    public static Instances train;
    public static Instances test;
    
    public static void main(String[] args){
        train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\HorseRacing\\HorseRacing_TRAIN");
        test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\HorseRacing\\HorseRacing_TEST");
        System.out.println("Num of train instances ="+train.numInstances());
        System.out.println("Num of test instances ="+test.numInstances());
        System.out.println("Num of attributes ="+(train.numAttributes()-1));
        int[] cDist=new int[2];
        for(int i=0;i<train.numInstances();i++)
            cDist[(int)(train.instance(i).classValue())]++;
        System.out.println("Train Losers ="+cDist[0]+" winners ="+cDist[1]);
       cDist=new int[2];
        for(int i=0;i<test.numInstances();i++)
            cDist[(int)(test.instance(i).classValue())]++;
        System.out.println("test Losers ="+cDist[0]+" test ="+cDist[1]);
        
        
        
        
    }
    
    
}
