package weka.classifiers.eas;


public class PSO2Classifier extends PSO1Classifier {



	public String globalInfo() {
		return "PSO con fitness 2 [De Falco, Della Cioppa, Tarantino, 2007].";
	}



	protected double evaluate(double[] x) {
		return fitness2(x);
	}



}

