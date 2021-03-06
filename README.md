# convert-caffemodel

The tutorial shows that how to convert caffemodel into tensorflow. It runs successfully in the environment:

* ubuntu 16.04 LTS
* cuda8.0 & cudnn6.0
* gcc-5.4



This conversion is required [Caffe](https://github.com/BVLC/caffe) and [MMdnn](https://github.com/Microsoft/MMdnn). Please check you have installed them, or follow the instructions to help you:

### How to install Caffe

1. **(optional)** In order to install happily, it's recommanded to use `Anaconda` environment when you are going to install Caffe. Follow [here](https://conda.io/docs/user-guide/install/linux.html) to install Anaconda, and create a environment `pycaffe` and activate it.

    ```sh
    # for python3 user
    conda create -n pycaffe python=3 anaconda
    # for python2 user
    conda create -n pycaffe python=2 anaconda
    
    source activate pycaffe
    ```

2. Get the lastest Caffe, and change directory:

    ```sh
    git clone https://github.com/BVLC/caffe ; cd caffe
    ```
    
3. Build it. In this step, you can select `Makefile` or `cmake` to do this task:

    * **Option-1:** `Makefile`
    
      a. Edit the Makefile.config. Uncomment lines you want to use. 
      ```sh
      cp Makefile.config.example Makefile.config
      vim Makefile.config
      # uncomment lines you want
      ```
      Or you can copy `Makefile.config` in this repo to the `caffe` folder:
      ```sh
      cp ../Makefile.config .
      ```

      b. Install required python packages. Make sure you are under the conda-env `pycaffe`:
      ```sh
      source activate pycaffe
      cd python
      for req in $(cat requirements.txt); do pip install $req; done
      cd ..
      ```

      c. Install python version, and assign the pycaffe path to PYTHONPATH.
      ```sh
      make pycaffe -j8
      export PYTHONPATH=/home/bass/data_server_storage/caffe/python:$PYTHONPATH
      ```

      d. Test your pycaffe.
      ```sh
      make pytest
      (...)
      python -c 'import caffe; print(caffe.__version__)
      ```
        
        
    * **Option-2:** `cmake` *(WARNING: may cause errors)*
    
      a. Deactivate the anaconda environment (we need to use system libraries). 
      ```sh
      source deactivate
      ```
      
      b. Create the `build` folder, and make it.
      ```sh
      mkdir build; cd build;
      cmake .. -DBLAS=open -Dpython_version=3 -Wno-dev
      source activate pycaffe
      make all install -j8
      ```
      If you are a python2 user, change to set `-Dpython_version=2`.
      If you do not have GPU, add the argument `-DCPU_ONLY=ON`.
      
      c. Install required python packages:
      ```sh
      cd ../python
      for req in $(cat requirements.txt); do pip install $req; done
      ```

      d. Install python version, and assign the pycaffe path to PYTHONPATH.
      ```sh
      make pycaffe -j8
      export PYTHONPATH=/home/bass/data_server_storage/caffe/python:$PYTHONPATH
      ```

      e. Test your pycaffe.
      ```sh
      make pytest
      (...)
      python -c 'import caffe; print(caffe.__version__)
      ```
    
    ### :fire: Errors & solutions:
    
    1. No module named 'google':

        ```
        ======================================================================
        ERROR: test_coord_map (unittest.loader._FailedTest)
        ----------------------------------------------------------------------
        ImportError: Failed to import test module: test_coord_map
        Traceback (most recent call last):
          File "/home/bass/anaconda3/lib/python3.6/unittest/loader.py", line 428, in _find_test_path
            module = self._get_module_from_name(name)
          File "/home/bass/anaconda3/lib/python3.6/unittest/loader.py", line 369, in _get_module_from_name
            __import__(name)
          File "/home/bass/data_server_storage/caffe/python/caffe/test/test_coord_map.py", line 6, in <module>
            import caffe
          File "/home/bass/data_server_storage/caffe/python/caffe/__init__.py", line 4, in <module>
            from .proto.caffe_pb2 import TRAIN, TEST
          File "/home/bass/data_server_storage/caffe/python/caffe/proto/caffe_pb2.py", line 6, in <module>
            from google.protobuf.internal import enum_type_wrapper
        ModuleNotFoundError: No module named 'google'
        ```
        :point_right: **solution**: `source activate pycaffe`
        
    2. `libboost-python` problem
    
        :point_right: **solution**: edit your `Makefile.config`:
        ```sh
        # (for python3 user) in Makefile.config
        --- PYTHON_LIBRARIES := boost_python python3.5m
        +++ PYTHON_LIBRARIES := boost_python-py35 python3.5m     
        ```
        Or change linkpaths `libboost_python.a` and `libboost_python.so` to direct to `libboos_python-py35`. If you use Anaconda, you can link them to your own anaconda library folder. For example:
        ```sh
        cd $HOME/anaconda3/lib/
        ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py35.a libboost_python3.a
        ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py35.so libboost_python3.so
        export LD_LIBRARY_PATH=$HOME/anaconda3/lib/:$LD_LIBRARY_PATH
        ```
        Make sure your anaconda library path is in `$LD_LIBRARY_PATH`.
        
    3. `hdf5` problem
    
        :point_right: **solution**: refer to [here](https://gist.github.com/wangruohui/679b05fcd1466bb0937f#hack-cuda-to-support-gcc-5)
        
    4. complier problems:
    
        ```sh
        In file included from ./include/caffe/common.hpp:6:0,
                 from ./include/caffe/blob.hpp:8,
                 from ./include/caffe/caffe.hpp:7,
                 from python/caffe/_caffe.cpp:17:
        ./include/caffe/net.hpp: In member function ‘const std::vector<caffe::Blob<Dtype>*>& caffe::Net<Dtype>::ForwardPrefilled(Dtype*)’:
        /usr/local/include/glog/logging.h:917:30: warning: typedef ‘INVALID_REQUESTED_LOG_SEVERITY’ locally defined but not used [-Wunused-local-typedefs]
                                      INVALID_REQUESTED_LOG_SEVERITY);           \
                                      ^
        /usr/local/include/glog/logging.h:912:73: note: in definition of macro ‘GOOGLE_GLOG_COMPILE_ASSERT’
           typedef google::glog_internal_namespace_::CompileAssert<(bool(expr))> msg[bool(expr) ? 1 : -1]
                                                                                 ^
        ./include/caffe/net.hpp:41:5: note: in expansion of macro ‘LOG_EVERY_N’
             LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
             ^
        ```
        :point_right: **solution**: make sure your gcc version is 5.4. `source deactivate`
    
    5. dynamic module does not define init function:
    
        ```sh
        ======================================================================
        ERROR: test_solver (unittest.loader.ModuleImportFailure)
        ----------------------------------------------------------------------
        ImportError: Failed to import test module: test_solver
        Traceback (most recent call last):
          File "/usr/lib/python2.7/unittest/loader.py", line 254, in _find_tests
            module = self._get_module_from_name(name)
          File "/usr/lib/python2.7/unittest/loader.py", line 232, in _get_module_from_name
            __import__(name)
          File "/home/bass/data_server_storage/caffe/python/caffe/test/test_solver.py", line 7, in <module>
            import caffe
          File "caffe/__init__.py", line 1, in <module>
            from .pycaffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, AdaDeltaSolver, AdamSolver, NCCL, Timer
          File "caffe/pycaffe.py", line 13, in <module>
            from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
        ImportError: dynamic module does not define init function (init_caffe)
        ```
        :point_right: **solution**: using `/usr/bin/gcc`. Another way that removes the original conda-env, and recreates it. 
        ```sh
        conda-env remove -n pycaffe ; conda-env create -n pycaffe python=3 anaconda
        ```
        
    6. `python-dateutil` version problem:
    
        ```sh
        pandas 0.22.0 has requirement python-dateutil>=2, but you'll have python-dateutil 1.5 which is incompatible.
        matplotlib 2.2.2 has requirement python-dateutil>=2.1, but you'll have python-dateutil 1.5 which is incompatible.
        ```
        :point_right: **solution**: `pip install --upgrade python-dateutil`
        
    7. No module named 'pydotplus':
    
        ```sh
        ======================================================================
        ERROR: test_draw (unittest.loader._FailedTest)
        ----------------------------------------------------------------------
        ImportError: Failed to import test module: test_draw
        Traceback (most recent call last):
          File "/home/bass/data_server_storage/caffe/python/caffe/draw.py", line 20, in <module>
            import pydotplus as pydot
        ModuleNotFoundError: No module named 'pydotplus'

        During handling of the above exception, another exception occurred:

        Traceback (most recent call last):
          File "/home/bass/anaconda3/envs/pycaffe/lib/python3.6/unittest/loader.py", line 428, in _find_test_path
            module = self._get_module_from_name(name)
          File "/home/bass/anaconda3/envs/pycaffe/lib/python3.6/unittest/loader.py", line 369, in _get_module_from_name
            __import__(name)
          File "/home/bass/data_server_storage/caffe/python/caffe/test/test_draw.py", line 6, in <module>
            import caffe.draw
          File "/home/bass/data_server_storage/caffe/python/caffe/draw.py", line 22, in <module>
            import pydot
        ModuleNotFoundError: No module named 'pydot'
        ```
        :point_right: **solution**: `pip install pydotplus`
        
    8. `glog/logging.h`:
    
        ```sh
        ...include/glog/logging.h(362): warning: using-declaration ignored -- it refers to the current namespace
        ```
        :point_right: **solution**: refer to https://github.com/PaddlePaddle/Paddle/issues/3277
        
    9. `protobuf` problems:
    
        ```sh
        lib/libcaffe.so: undefined reference to `google::collect2: error: ld returned 1 exit status
        ...
        .build_release/lib/libcaffe.so: undefined reference to `google::protobuf::Message::InitializationErrorString() const'
        ```
        :point_right: **solution**: Remove the original conda-env, and recreate it. 
        ```sh
        conda-env remove -n pycaffe ; conda-env create -n pycaffe python=3 anaconda
        ```
        
> If you failed to install Caffe with above steps, try to `make clean` and then install the corresponding packages with `conda` before `make`. Or follow [official installation instructions](http://caffe.berkeleyvision.org/installation.html) to install from beginning.


### How to install MMdnn

Follow [here](https://github.com/Microsoft/MMdnn#installation) to install MMdnn.

---

## Convert caffemodel

1. Prepare your environment.
```sh
export PYTHONPATH=/home/bass/data_server_storage/caffe/python:$PYTHONPATH
# for anaconda user
source activate tensorflow 
```

2. Download your own caffemodel and prototxt. For example, FCN8s:
```sh
wget https://raw.githubusercontent.com/simonguist/testing-fcn-for-cityscapes/master/final_model_url.txt
# download the caffemodel from above url

wget https://raw.githubusercontent.com/simonguist/testing-fcn-for-cityscapes/master/train/deploy_8s.prototxt
```

3. Using MMdnn to convert, for example, into tensorflow.
```sh
mmconvert -sf caffe -in deploy_8s.prototxt -iw cityscapes-fcn8s-2x.caffemodel -df tensorflow -om cityscapes-fcn8s-2x
```

4. Finally, you get the checkpoint named `cityscapes-fcn8s-2x`, and some random-named files including `.pd`, `.py`, `.json`, `.npy`.  You can rename and save them if you need.
  
