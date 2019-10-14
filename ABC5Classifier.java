package weka.classifiers.eas;


public class ABC5Classifier extends ABC1Classifier {



	protected double calculateFunction(double sol[]) {
		return fitness5(sol);
	}



	public String globalInfo() {
		return "ABC con fitness 5 [Karaboga, Ozturk, 2011].";
	}



}

