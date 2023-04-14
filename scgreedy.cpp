#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>


using Matrix = std::vector<std::vector<int>>;


struct Instance
{
	int nrows = 0;
	int ncols = 0;
	std::vector<int> obj;
	Matrix cols;
};


Instance readInstance(const char* filename)
{
	std::ifstream in(filename);
	Instance inst;
	in >> inst.nrows >> inst.ncols;
	assert(inst.nrows >= 1);
	assert(inst.ncols >= 1);
	for (int j = 0; j < inst.ncols; j++) {
		int cj = 0;
		int count = 0;
		in >> cj >> count;
		assert(cj > 0);
		assert(count > 0);
		inst.obj.push_back(cj);
		std::vector<int> col;
		for (int k = 0; k < count; k++) {
			int elem = 0;
			in >> elem;
			assert(1 <= elem && elem <= inst.nrows);
			elem--; //< we use 0-indexing, while file format uses 1-indexing
			col.push_back(elem);
		}
		inst.cols.emplace_back(col);
	}
	return inst;
}


Matrix transposeMatrix(const Instance& inst) {
	Matrix transposed(inst.nrows);
	for (int j = 0; j < inst.ncols; j++) {
		const auto& col = inst.cols[j];
		for (int i: col)  transposed[i].push_back(j);
	}
	return transposed;
}


using Solution = std::vector<int>;


Solution setcoverGreedy(const Instance& inst)
{
	int n = inst.ncols;
	int m = inst.nrows;
	Matrix rows = transposeMatrix(inst);
	assert(inst.ncols == inst.cols.size());
	assert(inst.nrows == rows.size());
	// How many times each row is covered by the current solution
	std::vector<int> rowcounts(m, 0);
	// How many (still uncovered rows) are covered by each column
	std::vector<int> colcounts(n, 0);
	for (int j = 0; j < n; j++)  colcounts[j] = inst.cols[j].size();
	Solution sol;
	// Greedy loop
	while (true) {
		double bestScore = 0.0;
		int bestJ = -1;
		// Look for best column
		for (int j = 0; j < n; j++) {
			double score = (double)colcounts[j] / (double)inst.obj[j];
			if (score > bestScore) {
				bestScore = score;
				bestJ = j;
			}
		}
		// Stop if nothing found (means that all rows are covered)
		if (bestJ == -1)  break;
		// Add best column to solution
		sol.push_back(bestJ);
		// Update row counts and col counts
		for (int i: inst.cols[bestJ]) {
			if (rowcounts[i] == 0) {
				for (int j: rows[i]) {
					colcounts[j]--;
					assert(colcounts[j] >= 0);
				}
			}
			rowcounts[i]++;
		}
	}
	return sol;
}


int evalObj(const Instance& inst, const Solution& sol)
{
	int tot = 0;
	for (int j: sol)  tot += inst.obj[j];
	return tot;
}


void writeSolution(const Solution& sol, int obj, const char* filename)
{
	std::ofstream out(filename);
	out << obj << std::endl;
	for (int j: sol)  out << j << " ";
}


int main (int argc, char const *argv[])
{
	Instance inst = readInstance(argv[1]);
	std::cout << "Read instance with " << inst.nrows << " rows and " << inst.ncols << " cols" << std::endl;
	Solution sol = setcoverGreedy(inst);
	int primalBound = evalObj(inst, sol);
	std::cout << "Found solution of value: " << primalBound << std::endl;
	writeSolution(sol, primalBound, argv[2]);
	return 0;
}