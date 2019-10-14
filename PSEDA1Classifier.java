package weka.classifiers.eas;

import java.util.Vector;
import org.apache.commons.math3.distribution.NormalDistribution;


public class PSEDA1Classifier extends EAClassifier {



	protected void makeCentroids() {
		//setto dimensionalita' (numero attributi x numero classi)
		D = nc*na;
		dim = D;
		//alloco
		x = new double[NP][D];
		p = new double[NP][D];
		g = new double[NP][D];
		fx = new double[NP];
		fp = new double[NP];
		fg = new double[NP];
		prex = new double[NP][D];
		prep = new double[NP][D];
		preg = new double[NP][D];
		prefx = new double[NP];
		prefp = new double[NP];
		prefg = new double[NP];
		lb = new double[D];
		ub = new double[D];
		n = new Vector[NP];
		for (int i=0; i<NP; i++)
			n[i] = new Vector<Integer>();
		best = new double[D];
		//setto bounds
		setBounds();
		//setto neighboroud type
		setNeighborhoodType();
		//setto pesi
		setWeights();
		//lancio algoritmo
		pseda();
		//copio best finale su centroids
		copyArraySolutionToMatrix(best,centroids);
		//disalloco
		x = null;
		p = null;
		g = null;
		fx = null;
		fp = null;
		fg = null;
		prex = null;
		prep = null;
		preg = null;
		prefx = null;
		prefp = null;
		prefg = null;
		lb = null;
		ub = null;
		n = null;
		best = null;
	}



	private void setBounds() {
		for (int i=0; i<D; i++) {
			lb[i] = minValues[i%na];
			ub[i] = maxValues[i%na];
		}
	}



	public String globalInfo() {
		return "PSEDA (global topology / variant 4) con wu=0.01 con fitness 1.";
	}



	protected double evaluate(double[] x) {
		return fitness1(x);
	}


	
	protected void setNeighborhoodType() {
		neighborhoodType = 2; //neighborhood type (1=ring, 2=global)
	}



	protected void setWeights() {
		wu = 0.01;
		wx = 0.09705;
		wp = 0.39795;
		wg = 0.39795;
		wpre = 0.09705;
	}



	int D, dim;//Dimensionality
	int NP = 50;//Population size
	int MAX_ITER = 1000;//Numero iterazioni
	int neighborhoodType;//neighborhood type (1=ring, 2=global)
	double wx,wp,wg,wu,wpre,wprex,wprep,wpreg;//pesi

	double[][] x;//positions
	double[][] p;//personal-best
	double[][] g;//neighborhood-best
	double[] fx;//fitness x
	double[] fp;//fitness p
	double[] fg;//fitness g
	double[][] prex;//previous positions
	double[][] prep;//previous personal-best
	double[][] preg;//previous neighborhood-best
	double[] prefx;//previous fitness x
	double[] prefp;//previous fitness p
	double[] prefg;//previous fitness g

	double[] lb;//bounds left
	double[] ub;//bounds right
	int ngen;
	double fbest;//best fitness attuale
	Vector<Integer>[] n;//neighborhood
	double stdev;//standard deviation
	double[] best;



	void setNeighborhood() {
		if (neighborhoodType==1) {//ring
			for (int i=0; i<NP; i++) {
				n[i].add(rem(i-1,NP));
				n[i].add(i);
				n[i].add(rem(i+1,NP));
			}
		} else if (neighborhoodType==2) {//global
			for (int i=0; i<NP; i++) {
				for (int j=0; j<NP; j++)
					n[i].add(j);
			}
		}
	}



	int rem(int a, int b) {
		if (a>=0)
			return a%b;
		else
			return b-1 - (-a-1)%b;
	}



	void copyDoubleArray(double a[],double b[],int d) {
		for (int i=0; i<d; i++)
			a[i] = b[i];
	}



	double urand01() {
		return rng.nextDouble();
	}




	void pseda() {
		//inizializzazioni varie
		ngen = 0;
		fbest = Double.POSITIVE_INFINITY;
		//Setto il vicinato
		setNeighborhood();
		//Calcolo wprex,wprep,wpreg
		wprex = wx/(wx+wp+wg);
		wprep = wp/(wx+wp+wg);
		wpreg = wg/(wx+wp+wg);
		//Inizializzazione particelle secondo pseda
		for (int i=0; i<NP; i++) {
			//x random
			for (int j=0; j<dim; j++) {
				x[i][j] = lb[j] + urand01()*(ub[j]-lb[j]);
			}
			//Calcolo fx
			fx[i] = evaluate(x[i]);
			//Personal best è inizializzato ad x,fx
			copyDoubleArray(p[i],x[i],dim);
			fp[i] = fx[i];
			//Inizializzo fg a +inf
			fg[i] = Double.POSITIVE_INFINITY;
			//prex e prep sono inizializzati a x e p
			copyDoubleArray(prex[i],x[i],dim);
			prefx[i] = fx[i];
			copyDoubleArray(prep[i],p[i],dim);
			prefp[i] = fp[i];
		}
		//MAIN-LOOP
		for (int iter=0; iter<MAX_ITER; iter++) {
			//Calcolo neighborhood best, preg, fbest
			for (int i=0; i<NP; i++) {
				//preg se non è la prima generazione
				if (ngen>0) {
					copyDoubleArray(preg[i],g[i],dim);
					prefg[i] = fg[i];
				}
				//neighborhood best
				for (int j=0; j<n[i].size(); j++) {
					if (fp[n[i].elementAt(j)]<fg[i]) {
						copyDoubleArray(g[i],p[n[i].elementAt(j)],dim);
						fg[i] = fp[n[i].elementAt(j)];
					}
				}
				//preg se è la prima generazione
				if (ngen==0) {
					copyDoubleArray(preg[i],g[i],dim);
					prefg[i] = fg[i];
				}
				//fbest
				if (fp[i]<fbest) {
					fbest = fp[i];
					copyDoubleArray(best,p[i],dim);
				}
			}
			//Incremento contatore generazioni
			ngen++;
			//Stampa info
			//print1();
			//Modifico posizione particelle
			for (int i=0; i<NP; i++) {
				//Prima di modificare x la salvo in temp
				double[] temp = new double[D];
				copyDoubleArray(temp,x[i],dim);
				//Modifico x calcolando anche stdev in base a distanza del centro verso l'altro centro più vicino
				for (int j=0; j<dim; j++) {
					//Due numeri casuali uniformi in [0,1]
					double r1 = urand01();
					double r2 = urand01();
					//In base ad r1 scelgo quale distr. componente della mistura campionare
					if (r1<=wu) {//campiono dall'uniforme
						x[i][j] = lb[j] + r2*(ub[j]-lb[j]);
					} else if (r1<=wu+wx) {//campiono dalla gaussiana troncata centrata in x
						double dxp = Math.abs(x[i][j]-p[i][j]);
						double dxg = Math.abs(x[i][j]-g[i][j]);
						if (x[i][j]==p[i][j] && x[i][j]==g[i][j])//dxp==0 && dxg==0
							stdev = 0.01*urand01();
						else if (x[i][j]==p[i][j])//dxp==0 && dxg!=0
							stdev = dxg;
						else if (x[i][j]==g[i][j])//dxp!=0 && dxg==0
							stdev = dxp;
						else//dxp!=0 && dxg!=0
							stdev = dxp<=dxg ? dxp : dxg;
						double cdfl = gaussian_cdf( (lb[j]-x[i][j])/stdev );
						double cdfu = gaussian_cdf( (ub[j]-x[i][j])/stdev );
						x[i][j] = gaussian_cdf_inv( cdfl+r2*(cdfu-cdfl) ) * stdev  +  x[i][j];
					} else if (r1<=wu+wx+wp) {//campiono dalla gaussiana troncata centrata in p
						double dpx = Math.abs(p[i][j]-x[i][j]);
						double dpg = Math.abs(p[i][j]-g[i][j]);
						if (p[i][j]==x[i][j] && p[i][j]==g[i][j])//dpx==0 && dpg==0
							stdev = 0.01*urand01();
						else if (p[i][j]==x[i][j])//dpx==0 && dpg!=0
							stdev = dpg;
						else if (p[i][j]==g[i][j])//dpx!=0 && dpg==0
							stdev = dpx;
						else//dpx!=0 && dpg!=0
							stdev = dpg<=dpx ? dpg : dpx;
						double cdfl = gaussian_cdf( (lb[j]-p[i][j])/stdev );
						double cdfu = gaussian_cdf( (ub[j]-p[i][j])/stdev );
						x[i][j] = gaussian_cdf_inv( cdfl+r2*(cdfu-cdfl) ) * stdev  +  p[i][j];
					} else if (r1<=wu+wx+wp+wg) {//campiono dalla gaussiana troncata centrata in g
						double dgx = Math.abs(g[i][j]-x[i][j]);
						double dgp = Math.abs(g[i][j]-p[i][j]);
						if (g[i][j]==x[i][j] && g[i][j]==p[i][j])//dgx==0 && dgp==0
							stdev = 0.01*urand01();
						else if (g[i][j]==x[i][j])//dgx==0 && dgp!=0
							stdev = dgp;
						else if (g[i][j]==p[i][j])//dgx!=0 && dgp==0
							stdev = dgx;
						else//dgx!=0 && dgp!=0
							stdev = dgp<=dgx ? dgp : dgx;
						double cdfl = gaussian_cdf( (lb[j]-g[i][j])/stdev );
						double cdfu = gaussian_cdf( (ub[j]-g[i][j])/stdev );
						x[i][j] = gaussian_cdf_inv( cdfl+r2*(cdfu-cdfl) ) * stdev  +  g[i][j];
					} else {//campiono dalla mistura precedente
						//serve un nuovo numero casuale uniforme in [0,1]
						double r3 = urand01();
						//In base ad r2 scelgo quale gaussiana campionare fra prex, prep, preg
						if (r2<=wprex) {//campiono prex
							double dxp = Math.abs(prex[i][j]-prep[i][j]);
							double dxg = Math.abs(prex[i][j]-preg[i][j]);
							if (prex[i][j]==prep[i][j] && prex[i][j]==preg[i][j])//dxp==0 && dxg==0
								stdev = 0.01*urand01();
							else if (prex[i][j]==prep[i][j])//dxp==0 && dxg!=0
								stdev = dxg;
							else if (prex[i][j]==preg[i][j])//dxp!=0 && dxg==0
								stdev = dxp;
							else//dxp!=0 && dxg!=0
								stdev = dxp<=dxg ? dxp : dxg;
							double cdfl = gaussian_cdf( (lb[j]-prex[i][j])/stdev );
							double cdfu = gaussian_cdf( (ub[j]-prex[i][j])/stdev );
							x[i][j] = gaussian_cdf_inv( cdfl+r3*(cdfu-cdfl) ) * stdev  +  prex[i][j];//nota qui c'è r3 e non r2 come sopra
						} else if (r2<=wprex+wprep) {//campiono prep
							double dpx = Math.abs(prep[i][j]-prex[i][j]);
							double dpg = Math.abs(prep[i][j]-preg[i][j]);
							if (prep[i][j]==prex[i][j] && prep[i][j]==preg[i][j])//dpx==0 && dpg==0
								stdev = 0.01*urand01();
							else if (prep[i][j]==prex[i][j])//dpx==0 && dpg!=0
								stdev = dpg;
							else if (prep[i][j]==preg[i][j])//dpx!=0 && dpg==0
								stdev = dpx;
							else//dpx!=0 && dpg!=0
								stdev = dpg<=dpx ? dpg : dpx;
							double cdfl = gaussian_cdf( (lb[j]-prep[i][j])/stdev );
							double cdfu = gaussian_cdf( (ub[j]-prep[i][j])/stdev );
							x[i][j] = gaussian_cdf_inv( cdfl+r3*(cdfu-cdfl) ) * stdev  +  prep[i][j];//nota qui c'è r3 e non r2 come sopra
						} else {//campiono preg
							double dgx = Math.abs(preg[i][j]-prex[i][j]);
							double dgp = Math.abs(preg[i][j]-prep[i][j]);
							if (preg[i][j]==prex[i][j] && preg[i][j]==prep[i][j])//dgx==0 && dgp==0
								stdev = 0.01*urand01();
							else if (preg[i][j]==prex[i][j])//dgx==0 && dgp!=0
								stdev = dgp;
							else if (preg[i][j]==prep[i][j])//dgx!=0 && dgp==0
								stdev = dgx;
							else//dgx!=0 && dgp!=0
								stdev = dgp<=dgx ? dgp : dgx;
							double cdfl = gaussian_cdf( (lb[j]-preg[i][j])/stdev );
							double cdfu = gaussian_cdf( (ub[j]-preg[i][j])/stdev );
							x[i][j] = gaussian_cdf_inv( cdfl+r3*(cdfu-cdfl) ) * stdev  +  preg[i][j];//nota qui c'è r3 e non r2 come sopra
						}
					}
				}
				//Calcolo nuova fitness
				fx[i] = evaluate(x[i]);
				//Metto temp in prex e p in prep
				copyDoubleArray(prex[i],temp,dim);
				copyDoubleArray(prep[i],p[i],dim);
				//Aggiorno best personale p
				if (fx[i] <= fp[i]) {
					copyDoubleArray(p[i],x[i],dim);
					fp[i] = fx[i];
				}
			}
			//Fine main-loop
		}
		//Stampa finale
		//print2();
	}



	NormalDistribution gaussian = null;



	double gaussian_cdf(double x) {
		//return gsl_cdf_ugaussian_P(x);
		if (gaussian==null)
			gaussian = new NormalDistribution();
		return gaussian.cumulativeProbability(x);
	}



	double gaussian_cdf_inv(double x) {
		//return gsl_cdf_ugaussian_Pinv(x);
		if (gaussian==null)
			gaussian = new NormalDistribution();
		return gaussian.inverseCumulativeProbability(x);
	}



}



