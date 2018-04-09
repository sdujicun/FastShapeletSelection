/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package applications;

import fileIO.InFile;
import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.*;

/**
 *
 * @author ajb
 */
public class FelixstowePort {
    public static String path = "C:\\Users\\ajb\\Dropbox\\Other Data\\";
    
   public static int getTowerIDS(String res){
       InFile all=new InFile(path+"felixstowe.csv");
       OutFile of=new OutFile(path+res);
       String header=all.readLine();
       String[] split=header.split(",");
       int count=0;
       for(String s:split){
           if(!s.equals("")){
            System.out.println(s);
            of.writeString(s+",");
            count++;
           }
       }
           System.out.println(" Size ="+split.length+" count ="+count);
           return count;
       
   } 
   public static SingleTower[] readData(String fullPath){
       InFile namesF=new InFile(path+"TowerID.csv");
       String[] names=namesF.readLine().split(",");
       int nosTowers=names.length;
        SingleTower[] all=new SingleTower[nosTowers];
        for(int i=0;i<nosTowers;i++){
            all[i]=new SingleTower(names[i]);
        }
        
        InFile f= new InFile(fullPath);
        int nosEntries=f.countLines()-3; //135135
        SingleTower.NOS_READINGS=nosEntries;
        System.out.println(" Number of towers ="+nosTowers+" Nos Readings ="+nosEntries);
        f= new InFile(fullPath);
        f.readLine();
        f.readLine();
        for(int i=0;i<nosEntries;i++){
            if(i%1000==0)
                System.out.println(" Reading line "+i);
            int[] time=new int[6];
//Day,Month,Year,Hour,Minute,Second            
            for(int j=0;j<6;j++)
                time[j]=f.readInt();
            Date d=new Date();
            d.setDate(time[0]);
            d.setMonth(time[1]);
            d.setYear(time[2]);
            d.setHours(time[3]);
            d.setMinutes(time[4]);
            d.setSeconds(time[5]);
            
            for(int j=0;j<nosTowers;j++){
                SingleTower.SingleEntry s =new SingleTower.SingleEntry();
                s.time=d;
//MsgType,Terminal,Windspeed,Gust,Direction,Alarm1,Alarm2                
                for(int k=0;k<7;k++){
                    s.data[k]=f.readDouble();
                    if(s.data[k]==-999.999)
                        s.missing=true;
                }        
                all[j].readings.add(s);
            }
        }
        return all;
       
   }
   public static void summariseStats(SingleTower[] a){
       for(int i=0;i<a.length;i++)
           a[i].findStats();
   }
   public static void saveStats(SingleTower[] a, String s){
       OutFile of= new OutFile(s);
       for(int i=0;i<a.length;i++)
           of.writeString(a[i]+"\n");
   }
   public static void findRangesAndDistributions(String path,SingleTower[] all){
       int fields=7;
       TreeSet[] data= new TreeSet[fields];
       for(int i=0;i<fields;i++)
           data[i]=new TreeSet<Double>();
      for(SingleTower a: all){
          for(SingleTower.SingleEntry s: a.readings)
            for(int i=0;i<fields;i++)
                data[i].add(s.data[i]);
      }
       TreeMap[] dist= new TreeMap[fields];
//Initialise counts       
      for(int i=0;i<fields;i++){
          dist[i]=new TreeMap<Double,IntWrapper>();
          Object[] a=data[i].toArray();
          for(int j=0;j<a.length;j++)
              dist[i].put(a[j],new IntWrapper());
      }
      for(SingleTower a: all){
          for(SingleTower.SingleEntry s: a.readings){
            for(int i=0;i<fields;i++){
                IntWrapper count=(IntWrapper)dist[i].get(s.data[i]);
                count.c++;    
            }           
          }
      }
      
      OutFile of=new OutFile(path);
      for(int i=0;i<fields;i++){
          of.writeString(SingleTower.SingleEntry.fieldNames[i]+",");
          for(Object d:data[i])
              of.writeString((Double)d+",");
          of.writeString("\n"+SingleTower.SingleEntry.fieldNames[i]+",");
          for(Object d:data[i]){
             IntWrapper count=(IntWrapper) dist[i].get(d);
             of.writeString(count.c+",");
          }
          of.writeString("\n");
      }

   }
   public static class IntWrapper{
       public int c=0;
   }
    public static void main(String[] args){
        System.out.println(" Loading Data...");
        SingleTower[] all=readData(path+"felixstowe.csv");
        System.out.println(" Finding Ranges...");
        findRangesAndDistributions(path+"fieldRanges.csv",all);
        System.out.println(" Finding Stats...");
        summariseStats(all);
        System.out.println(" Saving Stats...");
        saveStats(all,path+"summaryStats.csv");
    }
    
    
    public static class SingleTower{
        static DecimalFormat df=new DecimalFormat("###.###");
        static int NOS_READINGS=135126;
        String id;
        ArrayList<SingleEntry> readings;
        TowerStats stats;
        public SingleTower(){
            readings=new ArrayList<>();
        }
        public SingleTower(String name){
            readings=new ArrayList<>();
            id=name;
        }
        
        public void findStats(){
            stats=new TowerStats();
            stats.countValid();
        }
        public String toString(){
            return id+","+stats.toString();
        }
        
        
        public class TowerStats{
            int validReadings;
            double proportionValid;
            public void countValid(){
                for(SingleEntry s:readings){
                if(!s.missing)
                    validReadings++;
                }        
                proportionValid= (double)validReadings/(double)NOS_READINGS;
            }
            public String toString(){
                return validReadings+","+df.format(proportionValid);
            }
        
    }
        public static class SingleEntry{
        public Date time;
        public         boolean missing; //True if any field is missing
        public static String[] fieldNames={"msgType","terminal","windspeed","gust","direction","alarm1","alarm2"};
        public double[] data=new double[7];
        


    }
    }  
}
