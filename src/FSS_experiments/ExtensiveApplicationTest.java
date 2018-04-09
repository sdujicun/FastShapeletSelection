package FSS_experiments;

import java.util.Date;
import tsc_algorithms.COTE_FSS.COTE_FSS;
import development.DataSets;

public class ExtensiveApplicationTest {
	public static void main(String[] args) throws Exception {

		
		String path = DataSets.problemPath;
		//String[] problems={"ChlorineConcentration", "Coffee","DiatomSizeReduction","ECG200","ECGFiveDays","Lightning7","MoteStrain","Symbols","SyntheticControl","Trace"};		
		String[] problems=DataSets.DSUsed;
		
		System.out.println(problems.length);
		for (int i = 0; i < problems.length; i++) {
			System.out.print(problems[i] + "\t");
			System.out.print(new Date()+"\t");
			System.out.println(new COTE_FSS().trainTestExample(path, problems[i]));

		}
	}

}
