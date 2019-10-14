package weka.classifiers.eas;


public class PSEDA5Classifier extends PSEDA1Classifier {



	public String globalInfo() {
		return "PSEDA (global topology / variant 4) con wu=0.01 con fitness 5.";
	}



	protected double evaluate(double[] x) {
		return fitness5(x);
	}



}

