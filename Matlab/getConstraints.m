function [low, high] = getConstraints(samples, C)
	P = length(samples);

	low = zeros(P, 1);
	high = C/P*ones(P, 1);
end