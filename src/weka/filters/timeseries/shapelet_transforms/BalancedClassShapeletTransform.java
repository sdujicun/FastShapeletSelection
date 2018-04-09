/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;
import weka.core.Instances;
import weka.core.shapelet.Shapelet;
import weka.filters.timeseries.shapelet_transforms.searchFuntions.ShapeletSearch;

/**
 *
 * @author raj09hxu
 */
public class BalancedClassShapeletTransform extends FullShapeletTransform
{
    protected Map<Double, ArrayList<Shapelet>> kShapeletsMap;
    
    /**
     * protected method for extracting k shapelets.
     *
     * @param data the data that the shapelets will be taken from
     * @return an ArrayList of FullShapeletTransform objects in order of their
     * fitness (by infoGain, seperationGap then shortest length)
     */
    @Override
    public ArrayList<Shapelet> findBestKShapeletsCache(Instances data){
        
        
        ArrayList<Shapelet> seriesShapelets;                                    // temp store of all shapelets for each time series
        //construct a map for our K-shapelets lists, on for each classVal.
        
        if(kShapeletsMap == null){
            kShapeletsMap = new TreeMap();
            for (int i=0; i < data.numClasses(); i++){
                kShapeletsMap.put((double)i, new ArrayList<Shapelet>());
            }
        }
        
        //found out how many we want in each sub list.
        int proportion = numShapelets/kShapeletsMap.keySet().size();
        
        //for all time series
        outputPrint("Processing data: ");

        int dataSize = data.numInstances();
        //for all possible time series.
        while(dataSet < dataSize)
        {
            outputPrint("data : " + dataSet);
            
            //get the Shapelets list based on the classValue of our current time series.
            kShapelets = kShapeletsMap.get(data.get(dataSet).classValue());

            //we only want to pass in the worstKShapelet if we've found K shapelets. but we only care about this class values worst one.
            //this is due to the way we represent each classes shapelets in the map.
            worstShapelet = kShapelets.size() == proportion ? kShapelets.get(kShapelets.size()-1) : null;

            //set the series we're working with.
            subseqDistance.setSeries(dataSet);
            //set the clas value of the series we're working with.
            classValue.setShapeletValue(data.get(dataSet));
            
            seriesShapelets = searchFunction.SearchForShapeletsInSeries(data.get(dataSet), new ShapeletSearch.ProcessCandidate(){
            @Override
            public Shapelet process(double[] candidate, int start, int length){
               return checkCandidate(candidate, start, length);
            }});

            Collections.sort(seriesShapelets, shapeletComparator);

            seriesShapelets = removeSelfSimilar(seriesShapelets);

            kShapelets = combine(proportion, kShapelets, seriesShapelets);
            
            //re-update the list because it's changed now. 
            kShapeletsMap.put(data.get(dataSet).classValue(), kShapelets);
            
            dataSet++;
            
            createSerialFile();
        }

        kShapelets = buildKShapeletsFromMap(kShapeletsMap);
        
        this.numShapelets = kShapelets.size();

        recordShapelets(kShapelets, this.ouputFileLocation);
        printShapelets(kShapelets);

        return kShapelets;
    }
       
    private ArrayList<Shapelet> buildKShapeletsFromMap(Map<Double, ArrayList<Shapelet>> kShapeletsMap)
    {
       ArrayList<Shapelet> kShapelets = new ArrayList<>();
       
       int numberOfClassVals = kShapeletsMap.keySet().size();
       int proportion = numShapelets/numberOfClassVals;
       
       
       Iterator<Shapelet> it;
       
       //all lists should be sorted.
       //go through the map and get the sub portion of best shapelets for the final list.
       for(ArrayList<Shapelet> list : kShapeletsMap.values())
       {
           int i=0;
           it = list.iterator();
           
           while(it.hasNext() && i++ <= proportion)
           {
               kShapelets.add(it.next());
           }
       }
       return kShapelets;
    }
}
