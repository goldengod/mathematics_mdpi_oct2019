package weka.classifiers.eas;


public class ABC4Classifier extends ABC1Classifier {



	protected double calculateFunction(double sol[]) {
		return fitness4(sol);
	}



	public String globalInfo() {
		return "ABC con fitness 4 [Karaboga, Ozturk, 2011].";
	}



}

