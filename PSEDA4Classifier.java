package weka.classifiers.eas;


public class PSEDA4Classifier extends PSEDA1Classifier {



	public String globalInfo() {
		return "PSEDA (global topology / variant 4) con wu=0.01 con fitness 4.";
	}



	protected double evaluate(double[] x) {
		return fitness4(x);
	}



}

