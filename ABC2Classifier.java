package weka.classifiers.eas;


public class ABC2Classifier extends ABC1Classifier {



	protected double calculateFunction(double sol[]) {
		return fitness2(sol);
		//double tmp = fitness2(sol);
		//return tmp*tmp;
	}



	public String globalInfo() {
		return "ABC con fitness 2 [Karaboga, Ozturk, 2011].";
	}



}

