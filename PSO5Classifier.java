package weka.classifiers.eas;


public class PSO5Classifier extends PSO1Classifier {



	public String globalInfo() {
		return "PSO con fitness 5 [De Falco, Della Cioppa, Tarantino, 2007].";
	}



	protected double evaluate(double[] x) {
		return fitness5(x);
	}



}

