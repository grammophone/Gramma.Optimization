function [Q, g] = getGoalSystem(samples)

	P = length(samples);

	Q = getGramMatrix(samples);
	
	g = -ones(P, 1);

end