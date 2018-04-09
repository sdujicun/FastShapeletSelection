package weka.filters.timeseries.shapelet_transforms.subclass.dist;

import java.util.Comparator;

public class DistCompare implements Comparator{

    @Override
    public int compare(Object arg0, Object arg1) {
        // TODO Auto-generated method stub
        Dist dist1 = (Dist) arg0;
        Dist dist2 = (Dist) arg1;
       
        
        return dist1.getDist() > dist2.getDist() ? 1 : -1; //按照时间的由小到大排列 
    }

}
