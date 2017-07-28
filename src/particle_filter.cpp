/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;

	num_particles = 100;

	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);

	for (int i = 0 ; i < num_particles ; i++) {
		Particle p;
		p.id = i;
		
		p.x = N_x(gen);
		p.y = N_y(gen);
		p.theta = N_theta(gen);
		p.weight = 1;
		particles.push_back(p);
		weights.push_back(1);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	std::normal_distribution<double> N_x(0, std_pos[0]);
	std::normal_distribution<double> N_y(0, std_pos[1]);
	std::normal_distribution<double> N_theta(0, std_pos[2]);

	for (int i = 0 ; i < num_particles ; i++) {
		if (fabs(yaw_rate) > 0.00001) {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate* delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}

		

		particles[i].x += N_x(gen);
		particles[i].y += N_y(gen);
		particles[i].theta += N_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	double dist;
	int id_nearest;

	
	for (int i = 0 ; i < observations.size() ; i++) {
		LandmarkObs obs = observations[i];
		for (int j = 0 ; j < predicted.size() ; j++) {
			LandmarkObs pre = predicted[j];

			double a = fabs(obs.x - pre.x);
			double b = fabs(obs.y - pre.y);
			double c = sqrt(a*a + b*b);

			if (j == 0) {
				dist = c;
				id_nearest = pre.id;
			}
			else if (c < dist) {
				dist = c;
				id_nearest = pre.id;
			}
		}
		observations[i].id = id_nearest;
	}



}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	const double landmark_std_x = std_landmark[0];
	const double landmark_std_y = std_landmark[1];

	const long double multiplier = 1.0/(2*M_PI*landmark_std_x*landmark_std_y);
	const double cov_x = pow(landmark_std_x, 2.0);
	const double cov_y = pow(landmark_std_y, 2.0);
	
	for (int i = 0 ; i < num_particles ; i++) {
		vector<LandmarkObs> map_landmarks_within_range;

		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		
		vector<LandmarkObs> observations_in_global_space;

		for(int j = 0 ; j < observations.size() ; j++) {
			LandmarkObs lm;
			LandmarkObs o = observations[j];
			lm.x = p_x + (o.x * cos(p_theta) - o.y*sin(p_theta));
			lm.y = p_y + (o.x * sin(p_theta) + o.y * cos(p_theta));
			lm.id = o.id;
			
			
			observations_in_global_space.push_back(lm);
		}
		
		for (int l = 0 ; l < map_landmarks.landmark_list.size() ; l++) {
			LandmarkObs lm;
			lm.x = map_landmarks.landmark_list[l].x_f;
			lm.y = map_landmarks.landmark_list[l].y_f;
			lm.id = map_landmarks.landmark_list[l].id_i;
			double a = fabs(lm.x - p_x);
			double b = fabs(lm.y - p_y);
			double c = sqrt(a*a + b*b);

			if (c < sensor_range) map_landmarks_within_range.push_back(lm);
		}

		

		
		// for every observation, associate with nearest map_landmark
		dataAssociation(map_landmarks_within_range, observations_in_global_space);

		// multivariate Gaussian probabilty density
		double particle_weight = 1.0;

		

		for (int k = 0 ; k < observations_in_global_space.size() ; k++) {

			LandmarkObs observation = observations_in_global_space[k];

			LandmarkObs predicted_landmark;
			
			predicted_landmark.x = map_landmarks.landmark_list[observation.id-1].x_f;
			predicted_landmark.y = map_landmarks.landmark_list[observation.id-1].y_f;
			predicted_landmark.id = map_landmarks.landmark_list[observation.id-1].id_i;
			
			



			particle_weight *= exp(-pow(observation.x - predicted_landmark.x, 2.0)/(2.0*cov_x) - pow(observation.y - predicted_landmark.y, 2.0)/(2.0*cov_y));
			particle_weight *= multiplier;
		}
		

		particles[i].weight = particle_weight;
		weights[i] = particle_weight;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;

	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> new_particles;

	for (int i = 0 ; i < num_particles ; i++) {
		new_particles.push_back(particles[distribution(gen)]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
