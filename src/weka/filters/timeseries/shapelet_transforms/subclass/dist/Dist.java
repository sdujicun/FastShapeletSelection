package weka.filters.timeseries.shapelet_transforms.subclass.dist;

public class Dist {
	private int instanceIndex;
	private double dist;
	
	public Dist() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Dist(int instanceIndex, double dist) {
		super();
		this.instanceIndex = instanceIndex;
		this.dist = dist;
	}

	public int getInstanceIndex() {
		return instanceIndex;
	}

	public void setInstanceIndex(int instanceIndex) {
		this.instanceIndex = instanceIndex;
	}

	public double getDist() {
		return dist;
	}

	public void setDist(double dist) {
		this.dist = dist;
	}
	

}
