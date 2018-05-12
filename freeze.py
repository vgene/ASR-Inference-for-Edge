import os
import tensorflow as tf
from dynamic_brnn import DBiRNN
from utils import dotdict

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(args, maxTimeSteps):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    model = DBiRNN(args, maxTimeSteps)

    #for op in model.graph.get_operations():
    #    print(op)
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(args.savedir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model_1.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=model.graph) as sess:
        # We import the meta graph in the current default Graph
        model.saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            ["stack"] # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    args = dict()
    args['model'] = 'DBiRNN'
    args['num_layer'] = 2
    args['activation'] = tf.tanh
    args['batch_size'] = 1
    args['num_hidden'] = 256
    args['num_feature'] = 39
    args['num_class'] = 29
    args['num_epochs'] = 1
    args['savedir'] = './models/04262030'
    args = dotdict(args)
    maxTimeSteps = 1640
    freeze_graph(args, maxTimeSteps)
