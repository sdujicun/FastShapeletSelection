package representation.plr;

import java.util.ArrayList;
import java.util.List;



public class PLR_EP {
	private int i;


	public int[]  choosePointIP(double[] data) {
		int[] point = new int[data.length];
		for (int i = 0; i < data.length; i++) {
			point[i] = 0;
		}
		
		point[0] = 1;
		point[data.length - 1] = 1;

		for (int i = 1; i < data.length - 1; i++) {
			if (data[i] >= data[i + 1] && data[i] > data[i - 1]) {
				point[i] = 1;
			}
			if (data[i] <= data[i + 1] && data[i] < data[i - 1]) {
				point[i] = 1;
			}
		}
		return point;
	}

	
	
	public int[] choosePointIPByNum(double[] data) 
	{
		int[] IP=choosePointIP(data);
		List lis=new ArrayList();
		for(int i=0;i<IP.length;i++){
			if(IP[i]==1)
				lis.add(IP[i]);
		}
		int[] IPIndex=new int[lis.size()];
		for(int i=0;i<lis.size();i++){
			IPIndex[i]=(int)lis.get(i);
		}
		
		return IPIndex;
	
	}
	
	
	public int findNextPoint(int[] point, int end) {
		for (int k = end + 1; k < point.length; k++) {
			if (1 == point[k]) {
				end = k;
				return end;
			}
		}
		return end;
	}
	

}
