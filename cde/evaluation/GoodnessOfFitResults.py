
class GoodnessOfFitResults:
  def __init__(self, single_results_list):
    assert len(single_results_list) > 0, "given single results list is empty"

    self.single_results = single_results_list


    #for i in single_results_list:







  def __len__(self):
    return 1



  def plot_metric(self, graph_dict, metric='hellinger_distance', simulator='EconDensity'):
    raise NotImplementedError