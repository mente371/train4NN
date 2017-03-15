//
//  modelesNN.swift
//  train4ANN
//
//  Created by istros2015 on 18/02/2017.
//  Copyright © 2017 Istros Anlagen sro. All rights reserved.
//

import Foundation

@objc class NN53n : NNd {
    
    internal convenience init (pop : Bool) {
        self.init(ni:5, nh:3, no:1, ipop : pop)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        self.pop = pop
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger, ipop:Bool) {
        super.init()
        self.nom = "nn53n"
        self.pop = ipop
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.9, b: 0.9)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-2.0, b: 2.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
        //      determineAddCoefs()
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    let fcfy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfy,coef: 30.0)
                    let vareps = item2.objectForKey("etafiVar.eps") as! Double
                    let tvareps = truncsigm(vareps,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, tvareps, tfcfy, lcap],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 4 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN53a : NNd {
    
    internal convenience init (pop : Bool) {
        self.init(ni:5, nh:3, no:1, ipop : pop)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        self.pop = pop
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger, ipop:Bool) {
        super.init()
        self.nom = "nn53a"
        self.pop = ipop
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.8, b: 0.8)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-0.9, b: 0.9)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
  //      determineAddCoefs()
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    let trc4 = item2.objectForKey("trc4") as! Double
                    let varroce = item2.objectForKey("etafiVar.roce") as! Double
                    let tvarroce = truncsigm(varroce,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, trc4, tvarroce, lcap],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 4 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN63f : NNd {
    
    internal convenience init (pop : Bool) {
        self.init(ni:6, nh:3, no:1, ipop : pop)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        self.pop = pop
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger, ipop:Bool) {
        super.init()
        self.nom = "nn63f"
        self.pop = ipop
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.2, b: 0.2)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-2.0, b: 2.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
   //     determineAddCoefs()
    }

    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        //    let add = "~/Finance/Aetius/exportPopLarge.plist"
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 90 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    //          let alev : Double = item2.objectForKey("etafiNY.alev") as! Double
                    //          let talev = truncsigm(alev, coef: 80.0)
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //          let itrc4 = item2.objectForKey("trc4") as! Double
                    let n1Score = item2.objectForKey("n1Score") as! Double
                    let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    //            let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 30.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, tfcfy, n1Score, zlatoScore, lcap],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 2 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 5 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN53c : NN {
    
    internal convenience override init () {
        self.init(ni:5, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "NN53c"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.6, b: 0.6)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-0.8, b: 0.8)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
        
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //            let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    let n2Score = item2.objectForKey("n2Score") as! Double
                    //      let implGeva = item2.objectForKey("implGeva") as! Double
               //     let trc4 = item2.objectForKey("trc4") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 30.0)
                    //         let potg : Double = item2.objectForKey("etafiMoy.potg") as! Double
                    //        let tpotg = truncsigm(potg,coef: 30.0)
                    //             let varroce = item2.objectForKey("etafiVar.roce") as! Double
                    //             let tvarroce = truncsigm(varroce,coef: 20.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, n2Score, tfcfy, lcap],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 4 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN63c : NN {
    
    internal convenience override init () {
        self.init(ni:6, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "NN63c"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.8, b: 0.8)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-1.0, b: 1.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
        
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //            let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    let n2Score = item2.objectForKey("n2Score") as! Double
                    //      let implGeva = item2.objectForKey("implGeva") as! Double
                    //     let trc4 = item2.objectForKey("trc4") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 30.0)
                    //         let potg : Double = item2.objectForKey("etafiMoy.potg") as! Double
                    //        let tpotg = truncsigm(potg,coef: 30.0)
                    let varroeci = item2.objectForKey("etafiVar.roeci") as! Double
                    let tvarroeci = truncsigm(varroeci,coef: 40.0)
                //    let vareps = item2.objectForKey("etafiVar.eps") as! Double
                //    let tvareps = truncsigm(vareps,coef: 40.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, lcap, n2Score, tfcfy, tvarroeci],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 4 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 5 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 2 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN63b : NN {
    
    internal convenience override init () {
        self.init(ni:6, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "NN63b"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-1.2, b: 1.2)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-2.0, b: 2.0)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
        
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    //     let igearing : Double = item2.objectForKey("etafiNY.gearing") as! Double
                    //            let ic : Double = item2.objectForKey("etafiNY.ic") as! Double
                    let alev : Double = item2.objectForKey("etafiNY.alev") as! Double
                    let talev = truncsigm(alev,coef: 80.0)
                    //            let levScore = item2.objectForKey("levScore") as! Double
                    //            let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    let n1Score = item2.objectForKey("n1Score") as! Double
                    //      let implGeva = item2.objectForKey("implGeva") as! Double
                    let trc4 = item2.objectForKey("trc4") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 30.0)
                    //         let potg : Double = item2.objectForKey("etafiMoy.potg") as! Double
                    //        let tpotg = truncsigm(potg,coef: 30.0)
                    //             let varroce = item2.objectForKey("etafiVar.roce") as! Double
                    //             let tvarroce = truncsigm(varroce,coef: 20.0)
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, talev, trc4, tfcfy, n1Score, lcap],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 1 {
                ninputs[j] = truncsigm(inputs[j], coef:80.0)
            }
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 5 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN73b : NN {
    
    internal convenience override init () {
        self.init(ni:7, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "nn73b"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.8, b: 0.8)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-0.9, b: 0.9)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
        
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        //       let add = "~/Finance/Aetius/exportPopLarge.plist"
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roe : Double = item2.objectForKey("etafiNY.roe") as! Double
                    //     let igearing : Double = item2.objectForKey("etafiNY.gearing") as! Double
                    //              let ic : Double = item2.objectForKey("etafiNY.ic") as! Double
                    let alev = item2.objectForKey("etafiNY.alev") as! Double
                    let talev = truncsigm(alev,coef: 80.0)
                    let vareps = item2.objectForKey("etafiVar.eps") as! Double
                    let tvareps = truncsigm(vareps,coef: 40.0)
                    let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    //     let n1Score = item2.objectForKey("n1Score") as! Double
                    let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //           let implGeva = item2.objectForKey("etafiNY.implGeva") as! Double
                    let trc4 = item2.objectForKey("trc4") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    //         let varroce = item2.objectForKey("etafiVar.roce") as! Double
                    //         let tvarroce = truncsigm(varroce,coef: 20.0)
                    let tfcfy = truncsigm(fcfyMoy,coef: 35.0)
                    let tpotg = truncsigm(potg,coef: 30.0)
                    let iar : Array<Array<Double>> = [[roe, tpotg, zlatoScore, talev, trc4, tfcfy, tvareps],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 1 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:80.0)
            }
            if j == 5 {
                ninputs[j] = truncsigm(inputs[j], coef:35.0)
            }
            if j == 6 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN73c : NN {
    
    internal convenience override init () {
        self.init(ni:7, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "nn73c"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.6, b: 0.6)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-0.8, b: 0.8)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    let varroeci = item2.objectForKey("etafiVar.roce") as! Double
                    let tvarroeci = truncsigm(varroeci,coef: 40.0)
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    let vareps = item2.objectForKey("etafiVar.eps") as! Double
                    let tvareps = truncsigm(vareps,coef: 40.0)
          //          let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    let n3Score = item2.objectForKey("n3Score") as! Double
     //               let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //     let tpotg = truncsigm(potg,coef: 30.0)
                    //           let implGeva = item2.objectForKey("etafiNY.implGeva") as! Double
     //               let trc4 = item2.objectForKey("trc4") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 30.0)
                    
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, tvarroeci, levScore2, tfcfy, n3Score, tvareps, lcap],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 1 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 3 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 5 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 6 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN73d : NN {
    
    internal convenience override init () {
        self.init(ni:7, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    internal init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "nn73d"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //    if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.6, b: 0.6)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-0.8, b: 0.8)
                //    print(self.wi[j][k])
            }
        }
        //   }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
            //        let varroeci = item2.objectForKey("etafiVar.roce") as! Double
           //         let tvarroeci = truncsigm(varroeci,coef: 40.0)
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    let vareps = item2.objectForKey("etafiVar.eps") as! Double
                    let tvareps = truncsigm(vareps,coef: 40.0)
                    let varcapex = item2.objectForKey("etafiVar.eps") as! Double
                    let tvarcapex = truncsigm(varcapex,coef: 40.0)
                    //          let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    let n2Score = item2.objectForKey("n2Score") as! Double
                    //               let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    //     let tpotg = truncsigm(potg,coef: 30.0)
                    //           let implGeva = item2.objectForKey("etafiNY.implGeva") as! Double
                    //               let trc4 = item2.objectForKey("trc4") as! Double
                    let fcfyMoy = item2.objectForKey("etafiMoy.fcfy") as! Double
                    let tfcfy = truncsigm(fcfyMoy,coef: 30.0)
                    
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let iar : Array<Array<Double>> = [[roeci, levScore2, tfcfy, n2Score, tvareps, tvarcapex, lcap],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 1 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 2 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 4 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 5 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 6 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN83b : NN {

    internal convenience override init () {
        self.init(ni:8, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    
    init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "NN83b"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //        if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.7, b: 0.7)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-1.0, b: 1.0)
                //    print(self.wi[j][k])
            }
        }
        //        }
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        //       let add = "~/Finance/Aetius/exportPopLarge.plist"
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    //        let dcf : Double = item2.objectForKey("etafiNY.dcf") as! Double
                    //        let tdcf = truncsigm(dcf, coef: 30.0)
                    //        let levScore = item2.objectForKey("levScore") as! Double
                    let alev = item2.objectForKey("etafiNY.alev") as! Double
                    let talev = truncsigm(alev,coef: 80.0)
                    let trc4 = item2.objectForKey("trc4") as! Double
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    //             let roic = item2.objectForKey("etafiNY.roic") as! Double
                    //             let troic = truncsigm(roic,coef: 50.0)
                    //                let n1Score = item2.objectForKey("n1Score") as! Double
                    let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    let tpotg = truncsigm(potg, coef: 30.0)
                    let capexVar = item2.objectForKey("etafiVar.capex") as! Double
                    let tcapex = truncsigm(capexVar,coef: 40.0)
                    let vScore = item2.objectForKey("etafiNY.vScore") as! Double
                    //               let implGeva = item2.objectForKey("etafiNY.implGeva") as! Double
                    let vareps = item2.objectForKey("etafiVar.eps") as! Double
                    let tvareps = truncsigm(vareps,coef: 40.0)
                    let iar : Array<Array<Double>> = [[roeci, tvareps, talev, trc4, tpotg, vScore, tcapex, lcap],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 1 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 2 {
                ninputs[j] = truncsigm(inputs[j], coef:80.0)
            }
            if j == 4 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 6 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 7 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}

@objc class NN93g : NN {
    
    internal override convenience init () {
        self.init(ni:9, nh:3, no:1)
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 1.0
        cb = 2.0
        decodeNN()
    }
    init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        super.init()
        self.nom = "nn93g"
        self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + ".plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        
        // number of input, hidden, and output nodes
        self.ni = ni+1 // +1 for bias node
        self.nh = nh
        self.no = no
        
        ca = 1.0
        cb = 2.0
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        //        if decodeNN() == false {
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                self.wi[i][j]=randomFunc(-0.7, b: 0.7)
                //   print(self.wi[i][j])
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-1.0, b: 1.0)
                //    print(self.wi[j][k])
            }
        }
        //        }
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
    }
    
    override func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        let varob = "etafiNY.pbr"
        var arob = Array<Double>()
        for item in dat! {
            let item2  = item as! NSDictionary
            let varob = item2.objectForKey(varob) as! Double
            arob.append(varob)
        }
        let dico = statMono(arob)
        let e = dico["moy"]
        let s = dico["ec"]
        var pat = Array<Array<Array<Double>>>()
        if dat!.count >= 100 {
            for item in dat! {
                let item2  = item as! NSDictionary
                let ivarob = item2.objectForKey(varob) as! Double
                let zob = zscore(ivarob, e : e!, s : s!)
                if (ivarob > 0.0)&&(abs(zob) < 3.5) {
                    let roeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
                    //        let dcf : Double = item2.objectForKey("etafiNY.dcf") as! Double
                    //        let tdcf = truncsigm(dcf, coef: 30.0)
                    let levScore2 = item2.objectForKey("levScore2") as! Double
                    //     let alev = item2.objectForKey("etafiNY.alev") as! Double
                    //    let talev = truncsigm(alev,coef: 80.0)
                    let trc4 = item2.objectForKey("trc4") as! Double
                    let cap = item2.objectForKey("cap") as! Double
                    let lcap = logcap(cap)
                    let roic = item2.objectForKey("etafiNY.roic") as! Double
                    let troic = truncsigm(roic,coef: 50.0)
                    //        let n1Score = item2.objectForKey("n1Score") as! Double
                    let potg = item2.objectForKey("etafiMoy.potg") as! Double
                    let tpotg = truncsigm(potg, coef: 30.0)
                    //    let capexVar = item2.objectForKey("etafiVar.capex") as! Double
                    //    let tcapex = truncsigm(capexVar,coef: 40.0)
                    let vScore = item2.objectForKey("etafiNY.vScore") as! Double
                    //  let implGeva = item2.objectForKey("etafiNY.implGeva") as! Double
                    //  let timplGeva = truncsigm(implGeva,coef: 5.0)
                    let zlatoScore = item2.objectForKey("zlatoScore") as! Double
                    //    let vareva = item2.objectForKey("etafiVar.eva") as! Double
                    //    let teva = truncsigm(vareva , coef: 30.0);
                    let vareps = item2.objectForKey("etafiVar.eps") as! Double
                    let tvareps = truncsigm(vareps,coef: 40.0)
                    let iar : Array<Array<Double>> = [[roeci, tvareps, zlatoScore, levScore2, trc4, tpotg, vScore, troic, lcap],[ivarob]]
                    pat.append(iar)
                }
            }
        }
        return pat
    }
    
    override func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            if j == 1 {
                ninputs[j] = truncsigm(inputs[j], coef:40.0)
            }
            if j == 5 {
                ninputs[j] = truncsigm(inputs[j], coef:30.0)
            }
            if j == 7 {
                ninputs[j] = truncsigm(inputs[j], coef:50.0)
            }
            if j == 8 {
                ninputs[j] = logcap(inputs[j])
            }
            ninputs[j] =  cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
}
