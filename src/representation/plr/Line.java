package representation.plr;

public class Line {
	private int begin;
	private int end;
	private double dist;
	private double distMax;
	private int pmax;
	private double weight;
	public Line() {
		super();
		this.dist = 0;
		this.distMax = 0;
		this.weight=0;
	}
	public Line(int begin, int end, double dist, double distMax, int pmax,double weight) {
		super();
		this.begin = begin;
		this.end = end;
		this.dist = dist;
		this.distMax = distMax;
		this.pmax = pmax;
		this.weight=weight;
	}

	public int getBegin() {
		return begin;
	}
	public void setBegin(int begin) {
		this.begin = begin;
	}
	public int getEnd() {
		return end;
	}
	public void setEnd(int end) {
		this.end = end;
	}
	public double getDist() {
		return dist;
	}
	public void setDist(double dist) {
		this.dist = dist;
	}
	public double getDistMax() {
		return distMax;
	}
	public void setDistMax(double distMax) {
		this.distMax = distMax;
	}
	public int getPmax() {
		return pmax;
	}
	public void setPmax(int pmax) {
		this.pmax = pmax;
	}
	public double getWeight() {
		return weight;
	}
	public void setWeight(double weight) {
		this.weight = weight;
	}
	
}
