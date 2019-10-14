package weka.classifiers.eas;


public class ABC3Classifier extends ABC1Classifier {



	protected double calculateFunction(double sol[]) {
		return fitness3(sol);
	}



	public String globalInfo() {
		return "ABC con fitness 3 [Karaboga, Ozturk, 2011].";
	}



}

