import unittest
import polyhedron_heuristics.heterogeneous_annealing
import platform_exports.task_tools
import numpy

class TestOverlapPure(unittest.TestCase):
    def test_overlap_simple_1_first(self):
        start1 = 0
        start2 = 100
        end1 = 200
        end2 = 300
        expected_overlap = 100

        test_cost_function = polyhedron_heuristics.heterogeneous_annealing.soft_cost_function(None, None, None, None)
        calculated_overlap = test_cost_function._overlapping_time(start1,start2, end1, end2)

        self.assertEqual(expected_overlap,calculated_overlap)

    def test_overlap_simple_2_first(self):
        start1 = 100
        start2 = 0
        end1 = 200
        end2 = 300
        expected_overlap = 100

        test_cost_function = polyhedron_heuristics.heterogeneous_annealing.soft_cost_function(None, None, None, None)
        calculated_overlap = test_cost_function._overlapping_time(start1,start2, end1, end2)

        self.assertEqual(expected_overlap,calculated_overlap)

    def test_overlap_one_wrap_1(self):
        max_runtime = 500
        start1 = max_runtime - 100
        start2 = max_runtime - 200
        end1 = 200
        end2 = max_runtime
        expected_overlap = 100

        test_cost_function = polyhedron_heuristics.heterogeneous_annealing.soft_cost_function(None, None, max_runtime=max_runtime)
        calculated_overlap = test_cost_function._overlapping_time(start1,start2, end1, end2)

        self.assertEqual(expected_overlap,calculated_overlap)

    def test_overlap_one_wrap_2(self):
        max_runtime = 500
        start1 = max_runtime - 100
        start2 = max_runtime - 200
        end1 = 200
        end2 = max_runtime
        expected_overlap = 100

        test_cost_function = polyhedron_heuristics.heterogeneous_annealing.soft_cost_function(None, None, max_runtime=max_runtime)
        calculated_overlap = test_cost_function._overlapping_time(start2,start1, end2, end1)

        self.assertEqual(expected_overlap,calculated_overlap)

    def test_overlap_two_wrap_1(self):
        max_runtime = 500
        start1 = max_runtime - 100
        start2 = max_runtime - 200
        end1 = 200
        end2 = 100
        expected_overlap = 200

        test_cost_function = polyhedron_heuristics.heterogeneous_annealing.soft_cost_function(None, None, max_runtime=max_runtime)
        calculated_overlap = test_cost_function._overlapping_time(start1,start2, end1, end2)

        self.assertEqual(expected_overlap,calculated_overlap)

    def test_overlap_two_wrap_2(self):
        max_runtime = 500
        start1 = max_runtime - 100
        start2 = max_runtime - 200
        end1 = 200
        end2 = 100
        expected_overlap = 200

        test_cost_function = polyhedron_heuristics.heterogeneous_annealing.soft_cost_function(None, None, max_runtime=max_runtime)
        calculated_overlap = test_cost_function._overlapping_time(start2,start1, end2, end1)

        self.assertEqual(expected_overlap,calculated_overlap)

class TestOverlapWithTask(unittest.TestCase):
    def setUp(self):
        self.task1 = platform_exports.task_tools.task("Test1",0,200,"./tast1")
        self.task2 = platform_exports.task_tools.task("Test2",1,100,"./tast2")
        self.total_runtime = 500
        self.cost_function = polyhedron_heuristics.heterogeneous_annealing.soft_cost_function([self.task1, self.task2],1,self.total_runtime)

    def test_overlap_simple(self):
        x_test = numpy.zeros(shape=[4])
        x_test[0] = 100
        x_test[1] = 100
        expected_overlap = 100
        calculated_overlap = self.cost_function.get_task_exec_overlap(x_test,self.task2,self.task1)
        
        self.assertEqual(expected_overlap,calculated_overlap)

    def test_overlap_one_wrap_1(self):
        x_test = numpy.zeros(shape=[4])
        x_test[0] = self.total_runtime - 50
        x_test[1] = self.total_runtime - self.task2.get_exec_time()
        expected_overlap = 50
        calculated_overlap1 = self.cost_function.get_task_exec_overlap(x_test,self.task2,self.task1)
        calculated_overlap2 = self.cost_function.get_task_exec_overlap(x_test,self.task1,self.task2)
        
        self.assertEqual(expected_overlap,calculated_overlap1,calculated_overlap2)

    def test_overlap_one_wrap_2(self):
        x_test = numpy.zeros(shape=[4])
        x_test[0] = 0
        x_test[1] = self.total_runtime - 50
        expected_overlap = self.task2.get_exec_time() - 50
        calculated_overlap1 = self.cost_function.get_task_exec_overlap(x_test,self.task2,self.task1)
        calculated_overlap2 = self.cost_function.get_task_exec_overlap(x_test,self.task1,self.task2)
        
        self.assertEqual(expected_overlap,calculated_overlap1,calculated_overlap2)

    def test_overlap_two_wrap(self):
        x_test = numpy.zeros(shape=[4])
        x_test[0] = self.total_runtime - 100
        x_test[1] = self.total_runtime - 50
        expected_overlap = 100
        calculated_overlap1 = self.cost_function.get_task_exec_overlap(x_test,self.task2,self.task1)
        calculated_overlap2 = self.cost_function.get_task_exec_overlap(x_test,self.task1,self.task2)
        
        self.assertEqual(expected_overlap,calculated_overlap1,calculated_overlap2)
    
    def test_overlap_units_not_same(self):
        x_test = numpy.zeros(shape=[4])
        x_test[0] = self.total_runtime - 100
        x_test[1] = self.total_runtime - 50
        x_test[2] = 1
        x_test[3] = 2

        expected_overlap = 0
        calculated_overlap1 = self.cost_function.get_task_exec_overlap(x_test,self.task2,self.task1)
        calculated_overlap2 = self.cost_function.get_task_exec_overlap(x_test,self.task1,self.task2)
        
        self.assertEqual(expected_overlap,calculated_overlap1,calculated_overlap2)

if __name__ == "__main__":
    unittest.main()