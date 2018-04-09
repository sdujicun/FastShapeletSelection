/*
 Threads the find best k shapelets method
 */
package weka.filters.timeseries.shapelet_transforms;

import java.util.*;
import weka.core.Instances;
import weka.core.shapelet.Shapelet;

/**
 *
 * @author ajb
 */
public class ThreadedShapeletTransform extends FullShapeletTransform implements Runnable
{

    public static int numThreads = 16;
    public int startInst, endInst;
    public static Instances sharedInstances;
    public static int numS;
    public static int minL;
    public static int maxL;
    ArrayList<Shapelet> candidates;

    ThreadedShapeletTransform(int s, int e)
    {
        startInst = s;
        endInst = e;
    }

    @Override
    public void run()
    {

        //Performs find k best shapelets         
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    /**
     * protected method for extracting k shapelets.
     *
     * @param data the data that the shapelets will be taken from
     * @return
     *
     */
    @Override
    public ArrayList<Shapelet> findBestKShapeletsCache(Instances data)
    {       
        sharedInstances = data;
        numS = numShapelets;
        minL = minShapeletLength;
        maxL = maxShapeletLength;
        ThreadedShapeletTransform[] splits = new ThreadedShapeletTransform[numThreads];
        //Set the start and end instances for each thread.
        int interval = sharedInstances.numInstances() / numThreads;
        int s = 0;
        for (int i = 0; i < numThreads - 1; i++)
        {
            splits[i] = new ThreadedShapeletTransform(s, s + interval);
            s += interval;
        }
        splits[numThreads - 1] = new ThreadedShapeletTransform(s, sharedInstances.numInstances());
        Thread[] threads = new Thread[numThreads];
        //Generate the candidate shapelets for each thread
        try
        {
            for (int i = 0; i < numThreads; i++)
            {
                threads[i] = new Thread(threads[i]);
                threads[i].start();
            }
            for (int i = 0; i < numThreads; i++)
            {
                threads[i].join();
            }
        }
        catch (InterruptedException e)
        {
            System.out.println(" Fatal Error, thread interrupted, exit " + e);
        }
        //Wait for finish, then merge them back together.
        ArrayList<Shapelet> all = mergeShapelets(splits);
        return null;
    }

    public static ArrayList<Shapelet> mergeShapelets(ThreadedShapeletTransform[] s)
    {
        //Get all shapelets
        TreeSet<Shapelet> all = new TreeSet<>();
        for (ThreadedShapeletTransform sh : s)
        {
            all.addAll(sh.candidates);
        }
        //Extract out the top k
        return null;
    }

}
