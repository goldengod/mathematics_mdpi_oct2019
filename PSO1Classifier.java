//modifiche rispetto a PSOSalerno1Classifier.java segnalate con MODIFICA!!!
package weka.classifiers.eas;


public class PSO1Classifier extends EAClassifier {

	protected final static int NPOP = 50; //dim popolazione
	protected final static int TMAX = 1000; //numero generazioni

	protected double vmax; //velocita' massima
	protected double vmin; //velocita' minima
	protected double c1;
	protected double c2;
	protected double omega;

	protected int dim;
	protected double[][] pos; //posizione
	protected double[][] vel; //velocita'
	protected double[][] pb; //personal best
	protected double[] posf; //fitness della posizione
	protected double[] pbf; //personal best fitness
	protected int[] nb; //neighbourhood best indexes
	protected int gb; //global best index
	protected int gen; //indice generazione


	protected void makeCentroids() {
		//setto dimensionalita' (numero attributi x numero classi)
		dim = nc*na;
		//alloco
		pos = new double[NPOP][dim];
		vel = new double[NPOP][dim];
		pb = new double[NPOP][dim];
		posf = new double[NPOP];
		pbf = new double[NPOP];
		nb = new int[NPOP];
		//init parametri
		initParameters();
		//inizializzo pso
		initPSO();
		//ciclo pso
		mainLoopPSO();
		//copio best finale su centroids
		copyArraySolutionToMatrix(pb[gb],centroids);
		//disalloco
		pos = null;
		vel = null;
		pb = null;
		posf = null;
		pbf = null;
		nb = null;
	}



	private void initPSO() {
		gen = 0;
		for (int i=0; i<NPOP; i++) {
			for (int j=0; j<dim; j++) {
				pos[i][j] = minValues[j%na] + rng.nextDouble() * (maxValues[j%na] - minValues[j%na]);
				//vel[i][j] = vmin + rng.nextDouble() * (vmax - vmin); //MODIFICA!!!
				//Come in Standard PSO 2006 (particleswarmcentral)... differenza fra due pos a caso (diviso due)
				vel[i][j] = ( (minValues[j%na]+rng.nextDouble()*(maxValues[j%na]-minValues[j%na])) - pos[i][j] ) / 2.0;
				pb[i][j] = pos[i][j];
			}
			posf[i] = evaluate(pos[i]);
			pbf[i] = posf[i];
		}
		updateNb();
		updateGb();
	}



	private void mainLoopPSO() {
		//una generazione e' gia' stata fatta
		for (int t=1; t<TMAX; t++) {
			gen = t;
			updateParameters();
			for (int i=0; i<NPOP; i++) {
				for (int j=0; j<dim; j++) {
					updateVelocity(i,j);
					pos[i][j] += vel[i][j];
					//if ( pos[i][j]<minValues[j%na] || pos[i][j]>maxValues[j%na] )
					//	pos[i][j] = minValues[j%na] + rng.nextDouble() * (maxValues[j%na]-minValues[j%na]);
					if ( pos[i][j]<minValues[j%na] )
						pos[i][j] = minValues[j%na];
					else if ( pos[i][j]>maxValues[j%na] )
						pos[i][j] = maxValues[j%na];
				}
				posf[i] = evaluate(pos[i]);
				if ( posf[i]<=pbf[i] ) {
					pbf[i] = posf[i];
					for (int j=0; j<dim; j++)
						pb[i][j] = pos[i][j];
				}
			}
			updateNb();
			updateGb();
			//System.out.println(gen+") "+pbf[gb]);
		}
	}



	public String globalInfo() {
		return "PSO con fitness 1 [De Falco, Della Cioppa, Tarantino, 2007].";
	}


	protected final static double OMEGA_MAX = 0.9;
	protected final static double OMEGA_MIN = 0.4;


	protected void initParameters() {
		c1 = 2.0;
		c2 = 2.0;
		omega = OMEGA_MAX;
		vmax = 0.05;
		vmin = -0.05;
	}



	protected double evaluate(double[] x) {
		return fitness1(x);
	}



	protected void updateNb() {
	}



	protected void updateGb() {
		gb = 0;
		for (int i=1; i<NPOP; i++) {
			if ( pbf[i]<pbf[gb] )
				gb = i;
		}
	}



	protected void updateParameters() {
		omega = OMEGA_MAX - ( (OMEGA_MAX-OMEGA_MIN) * ((double)gen/(double)TMAX) );
	}



	protected void updateVelocity(int i,int j) {
		vel[i][j] = omega*vel[i][j] + rng.nextDouble()*c1*(pb[i][j]-pos[i][j]) + rng.nextDouble()*c2*(pb[gb][j]-pos[i][j]);
		//MODIFICA!!!
		//if ( vel[i][j]<vmin )
		//	vel[i][j] = vmin;
		//if ( vel[i][j]>vmax )
		//	vel[i][j] = vmax;
		//vmin e vmax trattate come percentuali della dim dello spazio di ricerca
		double bmin = vmin*(maxValues[j%na]-minValues[j%na]);
		double bmax = vmax*(maxValues[j%na]-minValues[j%na]);
		if ( vel[i][j]<bmin )
			vel[i][j] = bmin;
		if ( vel[i][j]>bmax )
			vel[i][j] = bmax;
	}



}



