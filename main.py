import sys
import os
import subprocess
import ntpath
from shutil import copyfile
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
import gui
import pybnf.parse as parse
import bnglparser as bngl

class bnfc(QtWidgets.QMainWindow, gui.Ui_mainWindow):
    '''
    Connects gui elements to functions
    '''
    varDict = {}
    paramDict = {}
    actionDict = {}
    mutantDict = {}
    models = {}
    paths = []
    varlist = []

    constructtypes = {}
    types = {}
    typenum = 0
    bnfpath = ""
    savepath = ""
    confhead = ""

    home = str(Path.home())

    def layout_widgets(self, layout):
        return (layout.itemAt(i).widget() for i in range(layout.count()))

    def __init__(self, parent=None):
        super(bnfc, self).__init__(parent)
        self.setupUi(self)
        #model config
        self.modelpBtn.clicked.connect(self.addModel)
        self.modelmBtn.clicked.connect(self.removeModel)
        self.exppBtn.clicked.connect(self.addExp)
        self.expmBtn.clicked.connect(self.removeExp)
        self.modelList.currentItemChanged.connect(self.modelItemChanged)
        #normalization
        self.typeaddBtn.clicked.connect(self.addType)
        self.typeremBtn.clicked.connect(self.removeType)
        self.expaddBtn.clicked.connect(self.addExp2)
        self.expremBtn.clicked.connect(self.removeExp2)
        self.typeList.currentItemChanged.connect(self.typeItemChanged)
        #file paths linked to text edits
        self.outputBtn.clicked.connect(self.outputdir)
        self.bngBtn.clicked.connect(self.bng_path)
        self.bnfpathBtn.clicked.connect(self.bnfpathfunc)
        #run bnf
        self.bnfBtn.clicked.connect(self.runBNF)
        #comboboxes
        self.initCb.activated[int].connect(self.initAct)
        self.fitCb.activated[int].connect(self.onActivated)
        self.objCb.activated[int].connect(self.objAct)
        self.cluster_type.activated[int].connect(self.clusterAct)

        self.actionOpen_file.triggered.connect(self.OpenFile)
        self.actionSave_file.triggered.connect(self.SaveFile)
        self.actionSave_Project.triggered.connect(self.SaveProject)
        self.openBtn.clicked.connect(self.OpenFile)
        self.saveBtn.clicked.connect(self.SaveFile)
        self.saveprojBtn.clicked.connect(self.SaveProject)

        #free params
        self.addBtn.clicked.connect(self.addEmptyParam)

        w = QtWidgets.QWidget()
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.playout = QtWidgets.QGridLayout()
        w.setLayout(self.playout)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(w)
        self.addEmptyParam()

        #time courses
        self.addCourseBtn.clicked.connect(self.addEmptyCourse)

        t = QtWidgets.QWidget()
        t.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.timeout = QtWidgets.QGridLayout()
        t.setLayout(self.timeout)
        self.timeArea.setWidgetResizable(True)
        self.timeArea.setWidget(t)
        self.addEmptyCourse()

        #param scan
        self.addScanBtn.clicked.connect(self.addEmptyScan)

        s = QtWidgets.QWidget()
        s.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.scanout = QtWidgets.QGridLayout()
        s.setLayout(self.scanout)
        self.scanArea.setWidgetResizable(True)
        self.scanArea.setWidget(s)
        self.addEmptyScan()

        #mutant
        self.addMutantBtn.clicked.connect(self.addEmptyMutant)

        m = QtWidgets.QWidget()
        m.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.mutantout = QtWidgets.QGridLayout()
        m.setLayout(self.mutantout)
        self.mutantArea.setWidgetResizable(True)
        self.mutantArea.setWidget(m)
        self.addEmptyMutant()

    '''
    model config functions
    '''
    def addModel(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "BioNetGen Files (*.bngl);;All Files (*)", options=options)
        if fileName != "":
            self.models[fileName] = []
            self.modelList.addItem(fileName)

    def removeModel(self):
        c = self.modelList.currentItem()
        if c is not None:
            self.modelList.takeItem(self.modelList.currentRow())
            self.models.pop(c.text(), None)

    def addExp(self):
        c = self.modelList.currentItem()
        if c is not None:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "QFileDialog.getOpenFileName()", "", "Experimental Data Files (*.exp, *.con);;Constraint Files (*.con);;All Files (*)", options=options)
            if fileName != "":
                self.models[c.text()].append(fileName)
                self.expList.addItem(fileName)

    def removeExp(self):
        m = self.modelList.currentItem()
        c = self.expList.currentItem()
        if c is not None:
            self.models[m.text()].remove(c.text())
            self.expList.takeItem(self.expList.currentRow())

    def modelItemChanged(self):
        c = self.modelList.currentItem()
        if c is not None:
            ml = self.models[c.text()]
            self.expList.clear()
            for i in ml:
                self.expList.addItem(i)
    '''
    normalization
    '''
    def addType(self):
        item = QtWidgets.QListWidgetItem()
        self.typenum += 1
        self.types[self.typenum] = []

        widget = QtWidgets.QWidget()
        widgetBox = QtWidgets.QComboBox()
        widgetBox.addItems(["init", "peak", "zero", "unit"])

        widgetLayout = QtWidgets.QHBoxLayout()
        widgetLayout.addWidget(widgetBox)

        widgetLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        widget.setLayout(widgetLayout)
        item.setSizeHint(widget.sizeHint())

        self.typeList.addItem(item)
        self.typeList.setItemWidget(item, widget)
        self.constructtypes[self.typenum] = widgetBox

    def addOpenType(self, index, exp):
        item = QtWidgets.QListWidgetItem()
        self.typenum += 1
        self.types[self.typenum] = []

        widget = QtWidgets.QWidget()
        widgetBox = QtWidgets.QComboBox()
        widgetBox.addItems(["init", "peak", "zero", "unit"])
        widgetBox.setCurrentIndex(index)

        widgetLayout = QtWidgets.QHBoxLayout()
        widgetLayout.addWidget(widgetBox)

        widgetLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        widget.setLayout(widgetLayout)
        item.setSizeHint(widget.sizeHint())

        self.typeList.addItem(item)
        self.typeList.setItemWidget(item, widget)
        self.constructtypes[self.typenum] = widgetBox

        self.types[self.typenum].append(exp)

    def removeType(self):
        c = self.typeList.currentItem()
        index = self.typeList.currentRow() + 1
        if c is not None:
            self.typeList.takeItem(self.typeList.currentRow())
            self.types.pop(self.types[index], None)

    def addExp2(self):
        c = self.typeList.currentItem()
        index = self.typeList.currentRow() + 1
        if c is not None:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "QFileDialog.getOpenFileName()", "", "Experimental Data Files (*.exp);;All Files (*)", options=options)
            if fileName != "":
                self.types[index].append(fileName)
                self.expList2.addItem(fileName)

    def removeExp2(self):
        m = self.typeList.currentItem()
        c = self.expList2.currentItem()
        index = self.typeList.currentRow() + 1
        if c is not None:
            self.types[index].remove(c.text())
            self.expList2.takeItem(self.expList.currentRow())

    def typeItemChanged(self):
        c = self.typeList.currentItem()
        index = self.typeList.currentRow() + 1
        if c is not None:
            ml = self.types[index]
            self.expList2.clear()
            for i in ml:
                self.expList2.addItem(i)

    '''
    file path funcs
    '''
    def outputdir(self):
        self.varDict["output_dir"] = str(
            QtWidgets.QFileDialog.getExistingDirectory(self, "Choose an output directory"))
        self.output_dir.setText(self.varDict.get("output_dir"))

    def bng_path(self):
        self.varDict["bng_command"] = str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Choose BioNetGen executable")[0])
        self.bng_command.setText(self.varDict.get("bng_command"))
    '''
    run BNF
    '''
    def bnfpathfunc(self):
        self.bnfpath = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose PyBioNetFit directory")
        self.bnf_path.setText(self.bnfpath)

    def runBNF(self):
        try:
            self.SaveFile()
            #print(self.savepath)
            command = str("cd " + self.bnfpath +"; pybnf -c " + self.savepath)
            QtWidgets.QMessageBox.about(self, "Alert", "Check terminal for BioNetFit subprocess")
            subprocess.Popen(command, shell=True, start_new_session=True)
        except Exception as e:
            print(e)
    '''
    init combobox functions
    '''
    def random(self):
        self.varDict["initialization"] = "random"

    def lh(self):
        self.varDict["initialization"] = "lh"

    def initAct(self, index):
        init = {0: self.random, 1: self.lh}
        init[index]()
    '''
    obj functions
    '''
    def chisq(self):
        self.varDict["objfunc"] = "chi_sq"

    def norm_sos(self):
        self.varDict["objfunc"] = "norm_sos"

    def ave_norm_sos(self):
        self.varDict["objfunc"] = "ave_norm_sos"

    def objAct(self, index):
        obj = {0: self.chisq, 1: self.norm_sos, 2: self.ave_norm_sos}
        obj[index]()
    '''
    cluster type cb
    '''
    def slurm(self):
        self.varDict["cluster_type"] = "slurm"

    def torque(self):
        self.varDict["cluster_type"] = "torque"

    def pbs(self):
        self.varDict["cluster_type"] = "pbs"

    def clusterAct(self, index):
        cltypes = {0: None, 1: self.slurm, 2: self.torque, 3: self.pbs}
        if index > 0:
            cltypes[index]()
    '''
    fit functions
    '''
    def de(self):
        self.varDict["fit_type"] = "de"
        self.fitStack.setCurrentIndex(0)

    def ade(self):
        self.varDict["fit_type"] = "ade"
        self.fitStack.setCurrentIndex(1)

    def pso(self):
        self.varDict["fit_type"] = "pso"
        self.fitStack.setCurrentIndex(2)

    def ss(self):
        self.varDict["fit_type"] = "ss"
        self.fitStack.setCurrentIndex(3)

    def bmc(self):
        self.varDict["fit_type"] = "bmc"
        self.fitStack.setCurrentIndex(4)

    def sim(self):
        self.varDict["fit_type"] = "sim"
        self.fitStack.setCurrentIndex(5)

    def sa(self):
        self.varDict["fit_type"] = "sa"
        self.fitStack.setCurrentIndex(6)

    def pt(self):
        self.varDict["fit_type"] = "pt"
        self.fitStack.setCurrentIndex(7)

    def dream(self):
        self.varDict["fit_type"] = "dream"
        self.fitStack.setCurrentIndex(8)

    def onActivated(self, index):
        fit = {0: self.de, 1: self.ade, 2: self.pso, 3: self.ss, 4: self.bmc, 5: self.sim, 6: self.sa, 7: self.pt, 8: self.dream}

        func = fit[index]
        func()

    '''
    utilities
    '''
    def varIterate(self, layout):
        for w in self.layout_widgets(layout):
            if type(w) == QtWidgets.QLineEdit:
                name = ''.join([i for i in w.objectName() if not i.isdigit()])
                if str(w.text()) != "":
                    self.varDict[str(name)] = str(w.text())
            elif type(w) == QtWidgets.QTextEdit:
                name = ''.join([i for i in w.objectName() if not i.isdigit()])
                if str(w.toPlainText()) != "":
                    self.varDict[str(name)] = str(w.toPlainText())
            elif type(w) == QtWidgets.QComboBox:
                name = ''.join([i for i in w.objectName() if not i.isdigit()])
                if name == "de_strategy":
                    r = {0: "rand1", 1: "rand2", 2: "best1", 3: "best2", 4: "all1", 5: "all2"}
                    self.varDict[str(name)] = r[w.currentIndex()]
    def clrIterate(self, layout):
        for w in self.layout_widgets(layout):
            if type(w) == QtWidgets.QLineEdit:
                w.setText("")
            elif type(w) == QtWidgets.QTextEdit:
                w.setPlainText("")

    def clearGrid(self, layout):
        if layout is not None:
            for w in self.layout_widgets(layout):
                w.deleteLater()

    '''
    free param func
    '''
    def min(self, label1, label2):
        label1.setText("min:")
        label2.setText("max:")

    def mu(self, label1, label2):
        label1.setText("mu:")
        label2.setText("sigma:")

    def step(self, label1, label2):
        label1.setText("init:")
        label2.setText("step:")

    def typeChange(self, ptype, index):
        ctype = {0: self.min, 1: self.mu, 2: self.min, 3: self.mu, 4: self.step, 5: self.step}
        for r in range(0, self.playout.rowCount()):
            it = self.playout.itemAtPosition(r, 1)
            if it is None:
                continue

            if it.widget() == ptype:
                label1 = self.playout.itemAtPosition(r, 2).widget()
                label2 = self.playout.itemAtPosition(r, 4).widget()
                ctype[index](label1, label2)
                break

    def addEmptyParam(self):
        name = QtWidgets.QLineEdit()
        ptype = QtWidgets.QComboBox()
        ptype.addItems(
            ["uniform var", "normal var", "log uniform var", "log normal var", "var -Simplex only", "log var -Simplex only"])

        slotLambda = lambda index: self.typeChange(ptype, index)
        ptype.activated[int].connect(slotLambda)
        val1 = QtWidgets.QLineEdit()
        val2 = QtWidgets.QLineEdit()

        gl = self.playout
        row = gl.rowCount()
        gl.addWidget(name, row, 0)
        gl.addWidget(ptype, row, 1)
        gl.addWidget(QtWidgets.QLabel("min:"), row, 2)
        gl.addWidget(val1, row, 3)
        gl.addWidget(QtWidgets.QLabel("max:"), row, 4)
        gl.addWidget(val2, row, 5)

    def addParam(self, typ, nme, v1, v2):
        name = QtWidgets.QLineEdit()
        ptype = QtWidgets.QComboBox()
        ptype.addItems(
            ["uniform var", "normal var", "log uniform var", "log normal var", "var -Simplex only", "log var -Simplex only"])

        slotLambda = lambda index: self.typeChange(ptype, index)
        ptype.activated[int].connect(slotLambda)
        val1 = QtWidgets.QLineEdit()
        val2 = QtWidgets.QLineEdit()

        gl = self.playout
        row = gl.rowCount()
        gl.addWidget(name, row, 0)
        gl.addWidget(ptype, row, 1)
        gl.addWidget(QtWidgets.QLabel("min:"), row, 2)
        gl.addWidget(val1, row, 3)
        gl.addWidget(QtWidgets.QLabel("max:"), row, 4)
        gl.addWidget(val2, row, 5)

        #assign values
        name.setText(nme)
        ptype.setCurrentIndex(typ)
        val1.setText(v1)
        val2.setText(v2)

    '''
    time courses func
    '''
    def addEmptyCourse(self):
        ctype = QtWidgets.QComboBox()
        ctype.addItems(["time", "step", "model", "suffix", "method"])

        val = QtWidgets.QLineEdit()

        gl = self.timeout
        row = gl.rowCount()
        gl.addWidget(QtWidgets.QLabel("key:"), row, 0)
        gl.addWidget(ctype, row, 1)
        gl.addWidget(QtWidgets.QLabel("val:"), row, 2)
        gl.addWidget(val, row, 3)

    def addCourse(self, typ, v1):
        ctype = QtWidgets.QComboBox()
        ctype.addItems(["time", "step", "model", "suffix", "method"])

        val = QtWidgets.QLineEdit()

        gl = self.timeout
        row = gl.rowCount()
        gl.addWidget(QtWidgets.QLabel("key:"), row, 0)
        gl.addWidget(ctype, row, 1)
        gl.addWidget(QtWidgets.QLabel("val:"), row, 2)
        gl.addWidget(val, row, 3)

        #assign values
        ctype.setCurrentIndex(typ)
        val.setText(v1)

    '''
    param scan
    '''
    def addEmptyScan(self):
        stype = QtWidgets.QComboBox()
        stype.addItems(["param:", "max:", "step:", "time:", "logspace:", "model:", "method:"])

        val = QtWidgets.QLineEdit()

        gl = self.scanout
        row = gl.rowCount()
        gl.addWidget(QtWidgets.QLabel("key:"), row, 0)
        gl.addWidget(stype, row, 1)
        gl.addWidget(QtWidgets.QLabel("val:"), row, 2)
        gl.addWidget(val, row, 3)

    def addScan(self, typ, v1):
        stype = QtWidgets.QComboBox()
        stype.addItems(["param:", "max:", "step:", "time:", "logspace:", "model:", "method:"])

        val = QtWidgets.QLineEdit()

        gl = self.scanout
        row = gl.rowCount()
        gl.addWidget(QtWidgets.QLabel("key:"), row, 0)
        gl.addWidget(stype, row, 1)
        gl.addWidget(QtWidgets.QLabel("val:"), row, 2)
        gl.addWidget(val, row, 3)

        #assign values
        stype.setCurrentIndex(typ)
        val.setText(v1)

    '''
    mutant
    '''
    def addMutantModel(self, model):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "BioNetGen Files (*.bngl);;All Files (*)", options=options)

        for r in range(0, self.mutantout.rowCount()):
            it = self.mutantout.itemAtPosition(r, 0)
            if it is None:
                continue

            if it.widget() == model:
                if fileName != "":
                    model.setText(fileName)
                break

    def addEmptyMutant(self):
        name = QtWidgets.QLineEdit()
        m = QtWidgets.QTextEdit()

        btn = QtWidgets.QPushButton("Choose:", None)
        slotLambda = lambda: self.addMutantModel(m)
        btn.clicked.connect(slotLambda)

        statement1 = QtWidgets.QLineEdit()
        statement2 = QtWidgets.QLineEdit()

        exp1 = QtWidgets.QLineEdit()
        exp2 = QtWidgets.QLineEdit()

        gl = self.mutantout
        row = gl.rowCount()
        gl.addWidget(m, row, 0)
        gl.addWidget(btn, row, 1)
        gl.addWidget(name, row, 2)
        gl.addWidget(statement1, row, 3)
        gl.addWidget(statement2, row, 4)
        gl.addWidget(QtWidgets.QLabel("|"), row, 5)
        gl.addWidget(exp1, row, 6)
        gl.addWidget(exp2, row, 7)

    def addMutant(self, mod, nam, stat1, stat2, ex1, ex2):
        name = QtWidgets.QLineEdit()
        m = QtWidgets.QTextEdit()

        btn = QtWidgets.QPushButton("+", None)
        slotLambda = lambda index: self.addMutantModel(m)
        btn.clicked.connect(slotLambda)

        statement1 = QtWidgets.QLineEdit()
        statement2 = QtWidgets.QLineEdit()

        exp1 = QtWidgets.QLineEdit()
        exp2 = QtWidgets.QLineEdit()

        gl = self.mutantout
        row = gl.rowCount()
        gl.addWidget(m, row, 0)
        gl.addWidget(btn, row, 1)
        gl.addWidget(name, row, 2)
        gl.addWidget(statement1, row, 3)
        gl.addWidget(statement2, row, 4)
        gl.addWidget(QtWidgets.QLabel("|"), row, 5)
        gl.addWidget(exp1, row, 6)
        gl.addWidget(exp2, row, 7)

        m.setText(mod)
        name.setText(nam)
        statement1.setText(stat1)
        statement2.setText(stat2)
        exp1.setText(ex1)
        exp2.setText(ex2)

    '''
    file menu func
    '''
    #copied load_config from parse.py, needed dict returned not Configuration object
    def load_config(self, path):
        try:
            infile = open(path, 'r')
        except FileNotFoundError:
            raise ValueError('Configuration file %s not found' % path)
        param_dict = parse.ploop(infile.readlines())
        infile.close()
        return param_dict

    def OpenFile(self):
        varDict = {}
        quickDict = {}

        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", "", "All Files (*);;Configure Files (*.conf)", options=options)
        if fileName != "":
            if not(fileName.endswith(".conf")):
                fileName += ".conf"
            self.savepath = fileName
            self.confhead = os.path.dirname(self.savepath)
            #print(self.confhead)
            self.current_file.setText(self.savepath)
        #clear everything if a file is opened while another file is in the gui
        if fileName != "" and bool(self.varDict):
            #init clear
            varDict = {}
            self.models = {}
            self.actionDict = {}
            self.paramDict = {}
            self.mutantDict = {}
                
            self.constructtypes = {}
            self.types = {}
            self.typenum = 0
            #model 
            self.modelList.clear()
            self.expList.clear()
            #mutant
            self.clearGrid(self.mutantout)
            #normalization
            self.typeList.clear()
            self.expList2.clear()
            #timecourses
            self.clearGrid(self.timeout)
            #param scan
            self.clearGrid(self.scanout)
            #free param
            self.clearGrid(self.playout)
            #algorithmic params
            self.clrIterate(self.de_layout)
            self.clrIterate(self.pso_layout)
            self.clrIterate(self.ss_layout)
            self.clrIterate(self.bmc_layout)
            self.clrIterate(self.sim_layout)
            self.clrIterate(self.sa_layout)
            self.clrIterate(self.pt_layout)
            self.clrIterate(self.dream_layout)
            #clear layouts
            self.clrIterate(self.go_layout)
            self.clrIterate(self.path_layout)
            self.clrIterate(self.cluster_layout)
            self.clrIterate(self.mult_layout)

        try:
            #set varDict to be the parsed dict that is returned
            varDict = self.load_config(fileName)
            #file paths
            self.output_dir.setPlainText(varDict.get("output_dir", "")) 
            self.bng_command.setPlainText(varDict.get("bng_command", ""))
            #iterate through the models key to find all the models and put them in the local models dictionary
            for m in varDict["models"]:
                exp = varDict[m]
                self.models[m] = exp
                self.modelList.addItem(m)

            #iterate through key value pairs to update gui
            for key, value in varDict.items():
                #print(key, value)
                if isinstance(key, tuple):
                    ptype = key[0]
                    name = key[1]
                    val1 = value[0]
                    val2 = value[1]
                    r = {"uniform_var": 0, "normal_var": 1, "loguniform_var": 2, "lognormal_var": 3}
                    self.addParam(r[ptype], name, str(val1), str(val2))
                elif key == "time_course":
                    quickDict = {}
                    for pair in value:
                        quickDict.update(pair)
                    for k, v in quickDict.items():
                        r = {"time": 0, "step": 1, "model": 2, "suffix": 3, "method": 4}
                        self.addCourse(r[k], v)
                elif key == "param_scan":
                    quickDict = {}
                    for pair in value:
                        quickDict.update(pair)
                    for k, v in quickDict.items():
                        r = {"param": 0, "max": 1, "step": 2, "time": 3, "logspace": 4, "model": 5, "method": 6}
                        self.addScan(r[k], v)
                elif key == "mutant":
                    self.addMutant(value[0], value[1], value[2], value[3], value[4], value[5], value[6])
                elif key == "normalization":
                    quickDict = {}
                   # print(value)
                    for k, v in value.items():
                        r = {"init": 0, "peak": 1, "zero": 2, "unit": 3}
                        self.addOpenType(r[v], k)

                else:
                    try:
                        lineEdit = self.findChild(QtWidgets.QLineEdit, key)
                        lineEdit.setText(str(value))
                    except:
                        pass

                #qcombobox setting
                #obj func
                obj = {"chi_sq": 0, "norm_sos": 1, "ave_norm_sos": 2}
                index = obj.get(varDict.get("objfunc"))
                if index is not(None):
                    self.objCb.setCurrentIndex(index)

                #init
                init = {"rand": 0, "lh": 1}
                index = init.get(varDict.get("initialization"))
                if index is not(None):
                    self.initCb.setCurrentIndex(index)

                #fit type
                fit = {"de": 0, "pso": 1, "ss": 2, "bmc": 3, "sim": 4, "sa": 5, "pt": 6}
                index = fit.get(varDict.get("fit_type"))
                if index is not(None):
                    self.fitCb.setCurrentIndex(index)
                    self.onActivated(index)

        except Exception as p:# parse.PybnfError as p:
            QtWidgets.QMessageBox.about(self, "Error:", "%s" % p)

    def Save(self, fileName):
        #store all values in varDict
        #file paths
        #stored in varDict in their own functions
        #styling for the output file
        self.savepath = fileName
        def coolComment(string):
            output = ""
            x = len(string) + 4
            i = 0
            while i < x:
                output += "#"
                i += 1
            output += str("\n# " + string + " #\n")
            j = 0
            while j < x:
                output += "#"
                j += 1
            output += "\n"
            return output
        #arrange variables in a pleasing manner
        def popDict(Dict, listItems):
            for key in listItems:
                if key in Dict:
                    file.write('%s = %s\n'%(key, Dict.get(key)))
                    Dict.pop(key, Dict.get(key))

        #general option params
        self.varIterate(self.go_layout)

        #cluster options
        self.varIterate(self.cluster_layout)
        i = {1: "slurm", 2: "torque", 3: "pbs"}
        if self.cluster_type.currentIndex() > 0:
            self.varDict["cluster_type"] = i[self.cluster_type.currentIndex()]

        #file paths
        self.varIterate(self.path_layout)

        #free params
        for r in range(0, self.playout.rowCount()):
            it = self.playout.itemAtPosition(r, 0)
            if it is not None:
                if str(it.widget().text()) != "":
                    val1 = self.playout.itemAtPosition(r, 3).widget()
                    val2 = self.playout.itemAtPosition(r, 5).widget()
                    index = self.playout.itemAtPosition(r, 1).widget().currentIndex()
                    r = {0: "uniform_var", 1: "normal_var", 2: "loguniform_var", 3: "lognormal_var"}
                    self.paramDict[str(it.widget().text())] = [r[index], val1.text(), val2.text()]

        #time courses
        self.actionDict["time_course"] = []
        for r in range(0, self.timeout.rowCount()):
            it = self.timeout.itemAtPosition(r, 3)
            if it is not None:
                if it.widget().text() != "":
                    val = self.timeout.itemAtPosition(r, 3).widget()
                    index = self.timeout.itemAtPosition(r, 1).widget().currentIndex()
                    r = {0: "time", 1: "step", 2: "model", 3: "suffix", 4: "method"}
                    l = [r[index], val.text()]
                    self.actionDict["time_course"].append(l)

        #param scan
        self.actionDict["param_scan"] = []
        for r in range(0, self.scanout.rowCount()):
            it = self.scanout.itemAtPosition(r, 3)
            if it is not None:
                if it.widget().text() != "":
                    val = self.scanout.itemAtPosition(r, 3).widget()
                    index = self.scanout.itemAtPosition(r, 1).widget().currentIndex()
                    r = {0: "param", 1: "max", 2: "step", 3: "time", 4: "logspace", 5: "model", 6: "method"}
                    l = [r[index], val.text()]
                    self.actionDict["param_scan"].append(l)

        #mutant
        for r in range(0, self.mutantout.rowCount()):
            it = self.mutantout.itemAtPosition(r, 0)
            if it is not(None):
                if it.widget().toPlainText() != "":
                    model = self.mutantout.itemAtPosition(r, 0).widget()
                    name = self.mutantout.itemAtPosition(r, 2).widget()
                    s1 = self.mutantout.itemAtPosition(r, 3).widget()
                    s2 = self.mutantout.itemAtPosition(r, 4).widget()
                    e1 = self.mutantout.itemAtPosition(r, 6).widget()
                    e2 = self.mutantout.itemAtPosition(r, 7).widget()
                    key = (model.toPlainText(), name.text(), s1.text(), s2.text())
                    self.mutantDict[key] = [e1.text(), e2.text()]

        #normalization
        normalDict = {}
        r = {0: "init", 1: "peak", 2: "zero", 3: "unit"}
        for k, v in self.constructtypes.items():
            typ = r[v.currentIndex()]
            normalDict.setdefault(typ, []).append(self.types.get(k))
       # print(normalDict)

        #algorithmic params
        self.varIterate(self.mult_layout)
        if not("initialization" in self.varDict):
            self.varDict["initialization"] = "random"

        #fit type
        #set default fit type to be de
        if not("fit_type" in  self.varDict):
            self.varDict["fit_type"] = "de"

        if self.varDict.get("refine") == "1":
            if self.varDict["fit_type"] == "de":
                self.varIterate(self.de_layout)
            elif self.varDict["fit_type"] == "pso":
                self.varIterate(self.pso_layout)
            elif self.varDict["fit_type"] == "ss":
                self.varIterate(self.ss_layout)
            elif self.varDict["fit_type"] == "bmc":
                self.varIterate(self.bmc_layout)
            elif self.varDict["fit_type"] == "sa":
                self.varIterate(self.sa_layout)
            elif self.varDict["fit_type"] == "pt":
                self.varIterate(self.pt_layout)
            else:
                self.varIterate(self.dream_layout)

        else:
            if self.varDict["fit_type"] == "de":
                self.varIterate(self.de_layout)
            elif self.varDict["fit_type"] == "pso":
                self.varIterate(self.pso_layout)
            elif self.varDict["fit_type"] == "ss":
                self.varIterate(self.ss_layout)
            elif self.varDict["fit_type"] == "bmc":
                self.varIterate(self.bmc_layout)
            elif self.varDict["fit_type"] == "sim":
                self.varIterate(self.sim_layout)
            elif self.varDict["fit_type"] == "sa":
                self.varIterate(self.sa_layout)
            elif self.varDict["fit_type"] == "pt":
                self.varIterate(self.pt_layout)
            else:
                self.varIterate(self.dream_layout)

        #obj func
        if not("objfunc" in self.varDict):
            self.varDict["objfunc"] = "chi_sq"

        if fileName != "":
            if not(fileName.endswith(".conf")):
                fileName += ".conf"
            self.current_file.setText(self.savepath)
            #print(self.savepath)
            paramAdj = {k: v for k, v in self.paramDict.items() if v is not None}
            with open(fileName, 'w') as file:
            #fundamental model specification
                #model map
                if self.models.items() != 0:
                    file.write(coolComment("Fundamental model specification"))
                    for key, value in self.models.items():
                        file.write('model = %s : '%(key) + str(", ".join(self.models[key])) + '\n')
                    popDict(self.varDict, ["bng_command", "output_dir", "fit_type", "objfunc", "parallel_count", "cluster_type", "scheduler_node", "worker_nodes"])
                #mutant
                for (model, name, s1, s2), (e1, e2) in self.mutantDict.items():
                    file.write("mutant = %s %s %s %s: %s %s\n"%(model, name, s1, s2, e1, e2))
                #action commands
                if self.actionDict.items() != 0:
                    file.write("\n")
                    for key, value in self.actionDict.items():
                        strprint = ""
                        if value:
                            file.write(key + " = ")
                            for k, v in value: #TEST FLAG
                                strprint += str(k + ": " + v + ",")
                            file.write(strprint[:-1])
                            file.write("\n")
            #parameter specification
                if paramAdj.items() != 0:
                    file.write(coolComment("Parameter Specifications"))
                    for name, (ptype, val1, val2) in paramAdj.items():
                        file.write('%s = %s %s %s\n'%(ptype, name, val1, val2))
                        self.varlist.append(name)
            #general options
                if self.varDict.items() != 0:
                    file.write(coolComment("General Options"))
                    for key, value in self.varDict.items():
                        file.write('%s = %s\n'%(key, value))
                if self.varDict.items() != 0:
                    file.write("#normalization\n")
                    for typ, exp in normalDict.items():
                        strprint = ""
                        file.write("normalization = " + typ + " : ") 
                        for k in exp:
                            for v in k:
                                strprint += str(v + ",")
                        strprint = str(strprint[:-1])
                        file.write(strprint)
                        file.write("\n")
    def SaveFile(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File","", "Configure Files (*.conf)", options=options)

        self.Save(fileName)

    def SaveProject(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        projectdir = QtWidgets.QFileDialog.getSaveFileName(
            self, "Choose an project directory", "", options=options)[0]

        if not os.path.isfile(projectdir):
            modellist = []
            explist = []

            for key, value in self.models.items(): #get the first model
                modellist.append(key)
                for val in value:
                    explist.append(val)

            if len(modellist) > 0:
                model = modellist[0]

            if not os.path.isdir(projectdir): #check if dir exists; create it
                os.makedirs(projectdir)

            confpath = str(projectdir + "/" + ntpath.basename(projectdir)) #conf path
            self.Save(confpath) #run save function for configure file

            #copy all exp files to new dir
            for exp in explist:
                if not(os.path.isfile(exp)):
                    npath = self.confhead + "/" + ntpath.basename(exp)
                    if os.path.isfile(npath):
                        exp = npath
                    else:
                        #print("Invalid exp path: "+ npath)
                        QtWidgets.QMessageBox.about(
                            self, "Alert", "One or more experimental file paths are invalid\nInvalid path: " + npath)
                        break
                export = projectdir+"/"+ntpath.basename(exp)
                try:
                    copyfile(exp, export)   
                except Exception as e:
                    QtWidgets.QMessageBox.about(self, "Alert", str(e))
                    break

            #copy all bngl files to new dir
            for model in modellist:
                if not(os.path.isfile(model)):
                    npath = self.confhead + "/" + ntpath.basename(model)
                    if os.path.isfile(npath):
                        model = npath
                    else:
                        QtWidgets.QMessageBox.about(
                            self, "Alert", "One or more model file paths are invalid")
                        break
                #set path for each model
                modelpath = projectdir +"/"+ntpath.basename(model)
                try:
                    #replace values of params with __FREE to make them free
                    #remove __FREE from name to find parameter
                    #print(model)
                    #print(modelpath)
                    #print(self.varlist)
                    bngl.load(model, modelpath, self.varlist)

                except Exception as e:
                    QtWidgets.QMessageBox.about(self, "Alert", str(e))
                    break
        else:
            QtWidgets.QMessageBox.about(
                            self, "Alert", "The path selected is not a valid directory")




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = bnfc()
    form.show()
    app.exec_()
