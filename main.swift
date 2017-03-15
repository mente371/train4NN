//
//  main.swift
//  train4ANN
//
//  Created by istros2015 on 18/02/2017.
//  Copyright © 2017 Istros Anlagen sro. All rights reserved.
//

import Foundation
import Swift

infix operator ** {}

func ** (num: Double, power: Double) -> Double{
    return pow(num, power)
}

infix operator *& {}
func *& (fill: Array<Double>, I: NSInteger) -> Array<Double>{
    var m = Array<Double>()
    let length = fill.count-1
    for _ in 1...I{
        for index in 0...length{
            m.append(fill[index])
        }
    }
    
    return m
}

func randomFunc(a: Double, b:Double) -> (Double) {
    //   let randNum = arc4random_uniform(100)/100
    let k : Int32 = rand()
    let max = RAND_MAX
    let f1 : Float = Float(k) / Float(max)
    
    let output = (b-a)*Double(f1) + (a)
    
    return output
}

func makeMatrix(I:NSInteger, J:NSInteger)->(Array<Array<Double>>){
    let NumColumns = I
    let NumRows = J
    var array = Array<Array<Double>>()
    for _ in 0...NumColumns-1 {
        array.append(Array(count:NumRows, repeatedValue:Double()))
    }
    
    return array
}

struct Result {
    var nom : String
    var mutvar : String
    var ddl : UInt32
    var npop : UInt
    var r2 : Float
    var corr : Float
    var median : Float
    var moy : Float
    init(noml : String) {
        self.nom = noml
        mutvar = String()
        ddl = 0
        npop = 0
        r2 = 0.0
        corr = 0.0
        median = 0.0
        moy = 0.0
    }
    func tr()->Float {
        if ddl > 0 {
            return Float(npop) / Float(ddl)
        }
        return 0.0
    }
}

protocol ModNN {
    func sigmoid(x: Double)->(Double)
    func dsigmoid(x: Double)->(Double)
    func logistic(x: Double)->(Double)
    func dlogistic(x: Double)->(Double)
    func truncsigm(x:Double, coef:Double)->Double
    func logcap (cap: Double)->(Double)
    func zscore(x : Double, e : Double, s : Double)->(Double)
    func statMono(ar : Array<Double>)->([String : Double])
    func calculSR (corr:Double, pop:UInt32)->(Double)
}

extension ModNN {
    func sigmoid(x: Double)->(Double){
        return tanh(x)
    }
    // derivative of our sigmoid function
    func dsigmoid(x: Double)->(Double){
        //  return 1.0 - x**2.0
        return 1.0 - pow(x,2.0)
    }
    func logistic(x: Double)->(Double){
        //    return tanh(x)
        return 1.0 / (1.0 + exp(-x))
    }
    // derivative of our sigmoid function
    func dlogistic(x: Double)->(Double){
        //  return 1.0 - x**2.0
        //  return 1.0 - pow(x,2.0)
        return x * (1.0 - x)
    }
    func truncsigm(x:Double, coef:Double)->Double {
        if coef > 0.0 {
            return coef * tanh(x/coef)
        } else {
            return 0.0
        }
    }
    func logcap (cap: Double)->(Double) {
        return log10(10.0 + (cap / 5.0))
    }
    func zscore(x : Double, e : Double, s : Double)->(Double) {
        if s > 0.0 {
            return (x - e) / s
        }
        return 0.0
    }
    func statMono(ar : Array<Double>)->([String : Double]) {
        var moy : Double = 0.0
        var sum : Double = 0.0
        var median : Double = 0.0
        var max : Double = 0.0
        var min : Double = 0.0
        var sx2 : Double = 0.0
        //     var var2 : Double = 0.0
        let n : Double = Double(ar.count)
        for ivar in ar {
            sum += ivar
            sx2 += (ivar * ivar)
            if ivar == ar[0] {
                max = ivar
                min = ivar
            }
            if ivar < min {
                min = ivar
            }
            max = (max < ivar) ? ivar : max
        }
        moy = sum / n
        /*
         for ivar in ar {
         var2 += (ivar - moy)*(ivar - moy)/n
         }
         */
        let vax = (sx2 / n) - (moy * moy)
        let sigx = sqrt(vax)
        let sar = ar.sort { $0 < $1 }
        let k = sar.count
        if k % 2 == 0 {
            median = (sar[k/2 - 1] + sar[k/2]) / 2.0
        } else {
            median = sar[k/2]
        }
        let dico = [ "moy" : moy,
                     "median" : median,
                     "min" : min,
                     "max" : max,
                     "ec" : sigx]
        return dico
    }
    func calculSR (corr:Double, pop:UInt32)->(Double) {
        var sr = 0.0
        if pop > 3 {
            if pop > 100 {
                sr = 1.96 / sqrt(Double(pop-1))
            }
            else {
                sr = tanh(1.96/sqrt(Double(pop-3)))
            }
        }
        return sr
    }
}

protocol Varmut {
    var metavar : String {get set}
    func truncsigm(x:Double, coef:Double)->Double
    func logcap (cap: Double)->(Double)
    func transmeta(imeta : Double)->(Double)
}
//sigmoid function. Later, will add more options for standard 1/(1+e^-x)

extension Varmut {
    func transmeta(imeta : Double)->(Double) {
        let itrans : Double
        switch metavar {
        case "etafiNY.roic":
            itrans = truncsigm(imeta,coef: 50.0)
        case "etafiNY.implGeva":
            itrans = truncsigm(imeta,coef: 10.0)
        case "etafiMoy.fcfy":
            itrans = truncsigm(imeta,coef: 30.0)
        case "etafiMoy.potg":
            itrans = truncsigm(imeta,coef: 30.0)
        case "etafiVar.eps":
            itrans = truncsigm(imeta,coef: 40.0)
        case "etafiVar.roce":
            itrans = truncsigm(imeta,coef: 40.0)
        case "etafiVar.capex":
            itrans = truncsigm(imeta,coef: 40.0)
        case "etafiVar.roeci":
            itrans = truncsigm(imeta,coef: 40.0)
        case "etafiVar.eva":
            itrans = truncsigm(imeta,coef: 40.0)
        case "cap":
            itrans = logcap(imeta)
        default:
            itrans = imeta
        }
        return itrans
    }
}

@objc class NN : NSObject, ModNN {
    var ni = 5
    var nh = 5
    var no = 1
    var ai = Array<Double>()
    var ah = Array<Double>()
    var ao = Array<Double>()
    var wi = Array<Array<Double>>()
    var wo = Array<Array<Double>>()
    var ci = Array<Array<Double>>()
    var co = Array<Array<Double>>()
    
    var minInputs = Array<Double>()
    var maxInputs = Array<Double>()
    var minOutputs = Array<Double>()
    var maxOutputs = Array<Double>()
    var addplistcoef : String?
    var addR2data : String?
    var nom : String = "NN"
    
    var r2 : Double = 0.0
    var corr : Double = 0.0
    var sr : Double = 0.0
    var npop : UInt32 = 0
    var moy : Double = 0.0
    var median : Double = 0.0
    
    var ca : Double = 0.0
    var cb : Double = 1.0
    
    func update(inputs:Array<Double>) -> (Array<Double>) {
        if (inputs.count != self.ni-1){
            print("wrong number of inputs")
        }
        
        for i in 0...(self.ni-2){
            //self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]
        }
        // hidden activations
        for j in 0...(self.nh-1){
            var sum = 0.0
            for i in 0...(self.ni-1){
                sum = sum + self.ai[i] * self.wi[i][j]
            }
            
            self.ah[j] = sigmoid(sum)
            
        }
        
        // output activations
        for k in 0...(self.no-1){
            var sum = 0.0
            for j in 0...(self.nh-1){
                sum = sum + self.ah[j] * self.wo[j][k]
            }
            
            self.ao[k] = sigmoid(sum)
        }
        
        return self.ao
    }
    
    func backPropagate(targets:Array<Double>, N:Double, M:Double)->(Double){
        if targets.count != self.no{
            print("wrong number of target values")
        }
        
        // calculate error terms for output
        var output_deltas = [0.0] *& self.no
        for k in 0...(self.no-1){
            let erroro = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * erroro
            //    print("outputerror :\(output_deltas[k])")
        }
        
        // calculate error terms for hidden
        var hidden_deltas = [0.0] *& self.nh
        for j in 0...(self.nh-1){
            var errorh = 0.0
            for k in 0...(self.no-1){
                errorh = errorh + output_deltas[k]*self.wo[j][k]
            }
            
            hidden_deltas[j] = dsigmoid(self.ah[j]) * errorh
        }
        
        // update output weights
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                let change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                //   print(N*change, M*self.co[j][k])
            }
        }
        
        // update input weights
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                let change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change
            }
        }
        
        // calculate error
        var error = 0.0
        for k in 0...(targets.count-1){
            error = error + 0.5*(targets[k]-self.ao[k])**2
        }
        
        return error
    }
    
    func normalise(patterns:Array<Array<Array<Double>>>)->(Array<Array<Array<Double>>>) {
        for p in 0...patterns.count-1{
            var inputs = patterns[p][0]
            var targets = patterns[p][1]
            if p == 0 {
                maxInputs = patterns[p][0]
                minInputs = patterns[p][0]
                maxOutputs = patterns[p][1]
                minOutputs = patterns[p][1]
            }
            for j in 0...inputs.count-1 {
                if inputs[j] > maxInputs[j] {
                    maxInputs[j] = inputs[j]
                } else
                    if inputs[j] < minInputs[j] {
                        minInputs[j] = inputs[j]
                }
            }
            for j in 0...targets.count-1 {
                if targets[j] > maxOutputs[j] {
                    maxOutputs[j] = targets[j]
                } else
                    if targets[j] < minOutputs[j] {
                        minOutputs[j] = targets[j]
                }
            }
        }
        let ni = patterns[0][0].count
        var maxminInputs = [Double](count: ni, repeatedValue:0.0)
        for j in 0...ni-1 {
            maxminInputs[j] = maxInputs[j] - minInputs[j]
        }
        let no = patterns[0][1].count
        var maxminOutputs = [Double](count: no, repeatedValue:0.0)
        for j in 0...no-1 {
            maxminOutputs[j] = maxOutputs[j] - minOutputs[j]
        }
        
        var npatterns = Array<Array<Array<Double>>>()
        
        for p in 0...patterns.count-1{
            var inputs = patterns[p][0]
            var targets = patterns[p][1]
            
            for j in 0...inputs.count-1 {
                inputs[j] =  cb * ((inputs[j] - minInputs[j]) / maxminInputs[j]) - ca
            }
            for j in 0...targets.count-1 {
                targets[j] =  cb * ((targets[j] - minOutputs[j]) / maxminOutputs[j]) - ca
            }
            npatterns.append([inputs,targets])
        }
        
        return npatterns
    }
    
    func majMaxMins (patterns:Array<Array<Array<Double>>>) {
        for p in 0...patterns.count-1{
            var inputs = patterns[p][0]
            var targets = patterns[p][1]
            if p == 0 {
                maxInputs = patterns[p][0]
                minInputs = patterns[p][0]
                maxOutputs = patterns[p][1]
                minOutputs = patterns[p][1]
            }
            for j in 0...inputs.count-1 {
                if inputs[j] > maxInputs[j] {
                    maxInputs[j] = inputs[j]
                } else
                    if inputs[j] < minInputs[j] {
                        minInputs[j] = inputs[j]
                }
            }
            for j in 0...targets.count-1 {
                if targets[j] > maxOutputs[j] {
                    maxOutputs[j] = targets[j]
                } else
                    if targets[j] < minOutputs[j] {
                        minOutputs[j] = targets[j]
                }
            }
        }
    }
    
    func normaliseInputsBruts(inputs:Array<Double>)->(Array<Double>) {
        let lni = inputs.count
        var ninputs = inputs//[Double](count: lni, repeatedValue:0.0)
        
        for j in 0...lni-1 {
            ninputs[j] = cb * ((ninputs[j] - minInputs[j]) / (maxInputs[j] - minInputs[j])) - ca
        }
        return ninputs
    }
    
    func encodeNN() {
        if let add = self.addplistcoef {
            let nadd = (add as NSString).stringByExpandingTildeInPath
            let arWi = self.wi as NSArray
            let arWo = self.wo as NSArray
            let metaAr : NSArray = [arWi, arWo] as NSArray
            metaAr.writeToFile(nadd, atomically: true)
            
            let add2 = self.applicationFilesDirectory()
            let filenameURL = NSURL(fileURLWithPath: add)
            let filename = filenameURL.lastPathComponent
            let newURL = add2.URLByAppendingPathComponent(filename!)
            metaAr.writeToURL(newURL, atomically: true)
        }
    }
    
    func applicationFilesDirectory()->(NSURL) {
        let fileManager = NSFileManager.defaultManager()
        let arrayAppSupportURL = fileManager.URLsForDirectory(.ApplicationSupportDirectory, inDomains:.UserDomainMask)
        let appSupportURL = arrayAppSupportURL.first
        let appSup2URL = appSupportURL!.URLByAppendingPathComponent("IA.Aetius36")
        return appSup2URL
    }
    
    func decodeNN()->(Bool) {
        if let add = self.addplistcoef {
            let nadd = (add as NSString).stringByExpandingTildeInPath
            let dat = NSArray(contentsOfFile:nadd)
            if let data = dat {
                if data.count == 2 {
                    let arWi = data[0] as! NSArray
                    let arWo = data[1] as! NSArray
                    self.wi = arWi as! Array<Array<Double>>
                    self.wo = arWo as! Array<Array<Double>>
                    return true
                }
            }
        }
        return false
    }
    
    func denorm(x:Double, min:Double, max:Double) -> (Double) {
        //   return min + ((x + 1.0) / 2.0 ) * (max - min)
        //      return min + (x * (max - min))
        return (max - min) * ((x + ca)/cb) + min
    }
    
    func targetpbr(inputs : Array<Double>)->(Array<Double>) {
        var aro = Array<Double>()
        for k in 0...no-1 {
            let mod = self.update(inputs)[k]
            let modn = denorm(mod, min: minOutputs[k], max: maxOutputs[k])
            aro.append(modn)
        }
        return aro
    }
    
    func weights()->(){
        print("weights:")
        for j in 0...(self.nh-1){
            for i in 0...(self.ni-1){
                print("we\(i).\(j):\(self.wi[i][j])")
            }
        }
        for k in 0...(self.no-1) {
            for j in 0...(self.nh-1){
                print("wo\(j).\(k):\(self.wo[j][k])")
            }
        }
    }
    
    func chargeData(add : String)->(Array<Array<Array<Double>>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        var pat = Array<Array<Array<Double>>>()
        
        for item in dat! {
            let item2  = item as! NSDictionary
            let ipbr = item2.objectForKey("etafiNY.pbr") as! Double
            let iroeci : Double = item2.objectForKey("etafiNY.roeci") as! Double
            let igearing : Double = item2.objectForKey("etafiNY.gearing") as! Double
            let ilevScore = item2.objectForKey("levScore") as! Double
            let itrc4 = item2.objectForKey("trc4") as! Double
            let ivarroce = item2.objectForKey("etafiVar.roce") as! Double
            let pivarroce = truncsigm(ivarroce,coef: 20.0)
            let iar : Array<Array<Double>> = [[iroeci, pivarroce, igearing, ilevScore, itrc4],[ipbr]]
            pat.append(iar)
        }
        return pat
    }
    
    func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger, N:Double=0.18, M:Double=0.005){
        // N: learning rate
        // M: momentum factor
        let npatterns = normalise(patterns)
        for i in 0...iterations{
            var error = 0.0
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                let targets = npatterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
                
            }
            // weights()
            if i % 1000 == 0{
                print("\(self.nom) error \(error)")
            }
        }
        encodeNN()
        let lr2 = calculeR2(npatterns)
        self.r2 = lr2
        print("\(self.nom) r2:\(lr2)")
        //    print("target pbr:\(targetpbr(npatterns[0][0]))")
    }
    
    func calculeR2 (patterns:Array<Array<Array<Double>>>)->(Double) {
        var arin = Array<Array<Double>>()
        for p in 0...patterns.count-1{
            let inputs = patterns[p][0]
            let targets = patterns[p][1]
            for k in 0...no-1 {
                let mod = self.update(inputs)[k]
                let modn = denorm(mod, min: minOutputs[k], max: maxOutputs[k])
                let reel = targets[k]
                let reeln = denorm(reel, min: minOutputs[k], max: maxOutputs[k])
                let ari = [modn, reeln]
                arin.append(ari)
            }
        }
        var r2 : Double = 0.0
        var fXi : Double = 0.0
        var fYi : Double = 0.0
        var SSres : Double = 0.0
        var SStot : Double = 0.0
        var fSX : Double = 0.0
        var fSY : Double = 0.0
        var fSX2 : Double = 0.0
        var fSY2 : Double = 0.0
        var fSXY : Double = 0.0
        var fYti : Double = 0.0
        var eY : Double = 0.0
        var h : UInt32 = 0
        var num : Double = 0.0
        var denom : Double = 0.0
        var corr : Double = 0.0
        var newar = Array<Array<Double>>()
        for i in 0...arin.count-1 {
            fXi = arin[i][0] // modn
            fYi = arin[i][1] // reeln
            let testX : Bool = (fXi > -99999.99) && (fXi < 99999.99)
            let testY : Bool = (fYi > -99999.99) && (fYi < 99999.99)
            let test : Bool = testX && testY
            if test {
                SSres += pow((fYi - fXi),2.0)
                fSY += fYi
                h += 1
                newar.append([fXi, fYi])
                fSX += fXi
                fSXY += (fXi * fYi)
                fSX2 += (fXi * fXi)
                fSY2 += (fYi * fYi)
            }
        }
        let dh = Double(h)
        if h != 0 {
            eY = fSY / dh
            num = fSXY - (fSX * fSY)/dh;
            denom = sqrt((fSX2-(fSX*fSX)/dh)*(fSY2-(fSY*fSY)/dh));
        }
        
        if (denom != 0.0) {
            corr = num / denom;
        }
        for j in 0...newar.count-1 {
            fYti = newar[j][1] - eY
            SStot += pow(fYti,2.0)
        }
        encodeR2data(newar)
        statModele(newar)
        if SStot > 0.0 {
            r2 = 1.0 - (SSres / SStot)
        }
        self.r2 = r2
        self.corr = corr
        self.npop = h
        self.sr = self.calculSR(corr, pop: h)
        return r2
    }
    
    func statModele(ar : Array<Array<Double>>) {
        var moy : Double = 0.0
        var sum : Double = 0.0
        var median : Double = 0.0
        var max : Double = 0.0
        var min : Double = 0.0
        var arrat = Array<Double>()
        for ari in ar {
            let mod = ari[0]
            let reel = ari[1]
            let ratio = mod/reel
            sum += ratio
            arrat.append(ratio)
            if ratio > max {
                max = ratio
            }
            if ratio < min {
                min = ratio
            }
        }
        moy = sum / Double(ar.count)
        let sarrat = arrat.sort { $0 < $1 }
        let k = sarrat.count
        if k % 2 == 0 {
            median = (sarrat[k/2 - 1] + sarrat[k/2]) / 2.0
        } else {
            median = sarrat[k/2]
        }
        self.moy = moy
        self.median = median
    }
    
    func calculeR2array (patterns:Array<Array<Array<Double>>>)->(Array<Double>) {
        var arin = Array<Array<Double>>()
        for p in 0...patterns.count-1{
            let inputs = patterns[p][0]
            let targets = patterns[p][1]
            for k in 0...0 {  //no-1
                let mod = self.update(inputs)[k]
                let modn = denorm(mod, min: minOutputs[k], max: maxOutputs[k])
                let reel = targets[k]
                let reeln = denorm(reel, min: minOutputs[k], max: maxOutputs[k])
                let ari = [modn, reeln]
                arin.append(ari)
            }
        }
        var r2 : Double = 0.0
        var fXi : Double = 0.0
        var fYi : Double = 0.0
        var SSres : Double = 0.0
        var SStot : Double = 0.0
        var fSX : Double = 0.0
        var fSY : Double = 0.0
        var fSX2 : Double = 0.0
        var fSY2 : Double = 0.0
        var fSXY : Double = 0.0
        var fYti : Double = 0.0
        var eY : Double = 0.0
        var h : UInt32 = 0
        var npos : UInt = 0
        var nposb : UInt = 0
        var num : Double = 0.0
        var denom : Double = 0.0
        var corr : Double = 0.0
        var newar = Array<Array<Double>>()
        for i in 0...arin.count-1 {
            fXi = arin[i][0] // modn
            fYi = arin[i][1] // reeln
            let np : UInt = fXi > 0.01 ? 1 : 0
            npos += np
            let npb : UInt = (fXi > 0.01)&&(fYi > 0.01) ? 1 : 0
            nposb += npb
            let testX : Bool = (fXi > -99999.99) && (fXi < 99999.99)
            let testY : Bool = (fYi > -99999.99) && (fYi < 99999.99)
            let test : Bool = testX && testY
            if test {
                SSres += pow((fYi - fXi),2.0)
                fSY += fYi
                h += 1
                newar.append([fXi, fYi])
                fSX += fXi
                fSXY += (fXi * fYi)
                fSX2 += (fXi * fXi)
                fSY2 += (fYi * fYi)
            }
        }
        let dh = Double(h)
        if h != 0 {
            eY = fSY / dh
            num = fSXY - (fSX * fSY)/dh;
            denom = sqrt((fSX2-(fSX*fSX)/dh)*(fSY2-(fSY*fSY)/dh));
        }
        if (denom != 0.0) {
            corr = num / denom;
        }
        for j in 0...newar.count-1 {
            fYti = newar[j][1] - eY
            SStot += pow(fYti,2.0)
        }
        if SStot > 0.0 {
            r2 = 1.0 - (SSres / SStot)
        }
        // encodeR2Test
        var text : String = String()
        text.appendContentsOf("model, reel\n")
        for i in 0...newar.count-1 {
            let ari = newar[i]
            text.appendContentsOf("\(ari[0]), \(ari[1])\n")
        }
        let addr2Test = "~/Finance/Aetius/r2TestnnX.csv"
        let nadd = (addr2Test as NSString).stringByExpandingTildeInPath
        let text2 = text as NSString
        try! text2.writeToFile(nadd, atomically: true, encoding: NSUnicodeStringEncoding)
        // fin encodeR2Test
      //  self.npos = npos
      //  self.nposb = nposb
        let lsr = self.calculSR(corr, pop: h)
        return [r2, corr, Double(h), lsr]
    }

    func validePopTest ()->(Array<Double>?) {
        var pat = Array<Array<Array<Double>>>()
        let add = "~/Finance/Aetius/exportPopTest.plist"
        pat = self.chargeData(add)
        let lni = pat[0][0].count
        if lni == self.ni-1 {
            let npatterns = normalise(pat)
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                self.update(inputs)
            }
            let ar = calculeR2array(npatterns)
            return ar
        }
        return nil
    }
    
    func batchtrain ()->() {
        var pat = Array<Array<Array<Double>>>()
        let add = "~/Finance/Aetius/exportPopLarge.plist"
        pat = self.chargeData(add)
        let lni = pat[0][0].count
        if lni == self.ni-1 {
            let ntrain : Int = self.ddl() * 20
            self.train(pat, iterations : ntrain)
        }
    }
    
    func batchtrainlight ()->() {
        var pat = Array<Array<Array<Double>>>()
        let add = "~/Finance/Aetius/exportPopLarge.plist"
        pat = self.chargeData(add)
        let lni = pat[0][0].count
        if lni == self.ni-1  {
            self.train(pat, iterations : 20)
            //    self.test(pat)
        }
    }
    
    func batchtrain (iterations : NSInteger)->() {
        var pat = Array<Array<Array<Double>>>()
        let add = "~/Finance/Aetius/exportPopLarge.plist"
        pat = self.chargeData(add)
        let lni = pat[0][0].count
        if lni == self.ni-1 {
            self.train(pat, iterations : iterations)
            //    self.test(pat)
        }
    }
    
    func test(patterns:Array<Array<Array<Double>>>)->(){
        let no = patterns[0][1].count
        //     for p in 0...patterns.count-1{
        for p in 0...0{
            for j in 0...no-1 {
                let retour : Array<Double> = self.update(patterns[p][0])
                print("\(patterns[p][0]) ->  \(denorm(retour[j], min:minOutputs[j], max:maxOutputs[j]))")
            }
        }
    }
    
    func encodeR2data(ar : Array<Array<Double>>) {
        var text : String = String()
        text.appendContentsOf("model, reel\n")
        for i in 0...ar.count-1 {
            let ari = ar[i]
            text.appendContentsOf("\(ari[0]), \(ari[1])\n")
        }
        let nadd = (self.addR2data! as NSString).stringByExpandingTildeInPath
        let text2 = text as NSString
        try! text2.writeToFile(nadd, atomically: true, encoding: NSUnicodeStringEncoding)
    }
    
    func ddl ()->(Int) {
        return nh * (ni + 1)
    }
    
    func ratiotrain ()->(Double) {
        return Double(npop) / Double(self.ddl())
    }
}

class NNd : NN {
    var pop : Bool = false
    var suf : String = "a"
    
    func determineAddCoefs ()->() {
        if self.pop {
            self.suf = "a"
            self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
            self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        } else {
            self.suf = "b"
            self.addplistcoef = "~/Finance/Aetius/dicoA" + nom + suf + ".plist"
            self.addR2data = "~/Finance/Aetius/testR2A" + nom + suf + ".csv"
        }
        decodeNN()
    }
    
    override func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger, N:Double=0.19, M:Double=0.007){
        // N: learning rate
        // M: momentum factor
        let npatterns = normalise(patterns)
        for i in 0...iterations{
            var error = 0.0
            for p in 0...npatterns.count-1{
                let inputs = npatterns[p][0]
                let targets = npatterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
                
            }
            if i % 500 == 0{
                print("\(self.nom) error \(error)")
            }
        }
        encodeNN()
        let lr2 = calculeR2(npatterns)
        self.r2 = lr2
        print("\(self.nom) r2:\(lr2)")
    }
    
    override func batchtrain (iterations : NSInteger)->() {
        var pat = Array<Array<Array<Double>>>()
        let add : String
        if self.pop == true {
            add = "~/Finance/Aetius/exportPopA.plist"
        } else {
            add = "~/Finance/Aetius/exportPopB.plist"
        }
        pat = self.chargeData(add)
        let lni = pat[0][0].count
        if lni == self.ni-1  {
            self.train(pat, iterations : iterations)
        }
        //   if let arTest = self.validePopTest() {
        //      print("\(self.nom) r2.Test:\(arTest[0])")
        //  }
    }
    
    override func batchtrainlight ()->() {
        var pat = Array<Array<Array<Double>>>()
        let add : String
        if self.pop == true {
            add = "~/Finance/Aetius/exportPopA.plist"
        } else {
            add = "~/Finance/Aetius/exportPopB.plist"
        }
        pat = self.chargeData(add)
        let lni = pat[0][0].count
        if lni == self.ni-1  {
            self.train(pat, iterations : 2)
            //    self.test(pat)
        }
    }
}

@objc class NNh : NN {
    override func normalise(patterns:Array<Array<Array<Double>>>)->(Array<Array<Array<Double>>>) {
        for p in 0...patterns.count-1{
            var inputs = patterns[p][0]
            var targets = patterns[p][1]
            if p == 0 {
                maxInputs = patterns[p][0]
                minInputs = patterns[p][0]
                maxOutputs = patterns[p][1]
                minOutputs = patterns[p][1]
            }
            for j in 0...inputs.count-1 {
                if inputs[j] > maxInputs[j] {
                    maxInputs[j] = inputs[j]
                } else
                    if inputs[j] < minInputs[j] {
                        minInputs[j] = inputs[j]
                }
            }
            for j in 0...targets.count-1 {
                if targets[j] > maxOutputs[j] {
                    maxOutputs[j] = targets[j]
                } else
                    if targets[j] < minOutputs[j] {
                        minOutputs[j] = targets[j]
                }
            }
        }
        let ni = patterns[0][0].count
        var maxminInputs = [Double](count: ni, repeatedValue:0.0)
        for j in 0...ni-1 {
            maxminInputs[j] = maxInputs[j] - minInputs[j]
        }
        let no = patterns[0][1].count
        var maxminOutputs = [Double](count: no, repeatedValue:0.0)
        for j in 0...no-1 {
            maxminOutputs[j] = maxOutputs[j] - minOutputs[j]
        }
        
        //       maxTransOutputs = [0.0]
        //        print("outputs bruts[0]: \(patterns[0][1])")
        var npatterns = Array<Array<Array<Double>>>()
        
        for p in 0...patterns.count-1{
            var inputs = patterns[p][0]
            var targets = patterns[p][1]
            let ca : Double = 1.0
            let cb : Double = 2.0
            
            for j in 0...inputs.count-1 {
                inputs[j] =  (inputs[j] - minInputs[j]) / maxminInputs[j]
            }
            for j in 0...targets.count-1 {
                let tj = acosh((targets[j] / minOutputs[j]))
                let tcj = maxOutputs[j]/minOutputs[j]
                targets[j] =  cb * (tj / acosh(tcj)) - ca
            }
            npatterns.append([inputs,targets])
        }
        //       print("outputs normés[0]: \(npatterns[0][1])")
        return npatterns
    }
    
    override func denorm(x:Double, min:Double, max:Double) -> (Double) {
        //   return min + ((x + 1.0) / 2.0 ) * (max - min)
  //      let ca = 1.0
  //      let cb = 2.0
        //      return min + (x * (max - min))
        // return (max - min) * ((x + ca)/cb) + min
        return cosh(acosh(max/min) * (x + ca)/cb) * min
    }
}

@objc class NNdh : NNd {
    override func normalise(patterns:Array<Array<Array<Double>>>)->(Array<Array<Array<Double>>>) {
        for p in 0...patterns.count-1{
            var inputs = patterns[p][0]
            var targets = patterns[p][1]
            if p == 0 {
                maxInputs = patterns[p][0]
                minInputs = patterns[p][0]
                maxOutputs = patterns[p][1]
                minOutputs = patterns[p][1]
            }
            for j in 0...inputs.count-1 {
                if inputs[j] > maxInputs[j] {
                    maxInputs[j] = inputs[j]
                } else
                    if inputs[j] < minInputs[j] {
                        minInputs[j] = inputs[j]
                }
            }
            for j in 0...targets.count-1 {
                if targets[j] > maxOutputs[j] {
                    maxOutputs[j] = targets[j]
                } else
                    if targets[j] < minOutputs[j] {
                        minOutputs[j] = targets[j]
                }
            }
        }
        let ni = patterns[0][0].count
        var maxminInputs = [Double](count: ni, repeatedValue:0.0)
        for j in 0...ni-1 {
            maxminInputs[j] = maxInputs[j] - minInputs[j]
        }
        let no = patterns[0][1].count
        var maxminOutputs = [Double](count: no, repeatedValue:0.0)
        for j in 0...no-1 {
            maxminOutputs[j] = maxOutputs[j] - minOutputs[j]
        }
        
        //       maxTransOutputs = [0.0]
        //        print("outputs bruts[0]: \(patterns[0][1])")
        var npatterns = Array<Array<Array<Double>>>()
        
        for p in 0...patterns.count-1{
            var inputs = patterns[p][0]
            var targets = patterns[p][1]
     //       let ca : Double = 1.0
     //       let cb : Double = 2.0
            
            for j in 0...inputs.count-1 {
                inputs[j] =  (inputs[j] - minInputs[j]) / maxminInputs[j]
            }
            for j in 0...targets.count-1 {
                let tj = acosh((targets[j] / minOutputs[j]))
                let tcj = maxOutputs[j]/minOutputs[j]
                targets[j] =  cb * (tj / acosh(tcj)) - ca
            }
            npatterns.append([inputs,targets])
        }
        //       print("outputs normés[0]: \(npatterns[0][1])")
        return npatterns
    }
    
    override func denorm(x:Double, min:Double, max:Double) -> (Double) {
        //   return min + ((x + 1.0) / 2.0 ) * (max - min)
  //      let ca = 1.0
  //     let cb = 2.0
        //      return min + (x * (max - min))
        // return (max - min) * ((x + ca)/cb) + min
        return cosh(acosh(max/min) * (x + ca)/cb) * min
    }
}

func train()->(){
     let n = NN63k()
//   let n = NN73k(ni:7, nh:3, no:1)
//    let n = NN63f(pop: false)
//    let n = NN53n(ni:5, nh:3, no:1, ipop: false)
//    n.determineAddCoefs()
    n.batchtrain(3000)
    print("ddl: \(n.ddl()), pop: \(n.npop), ratiotrain: \(n.ratiotrain()), corr: \(n.corr)\npbr modele/reel: moy:\(n.moy), median:\(n.median)")
}

var armet = ["trc4", "n1Score", "n3Score", "zlatoScore", "etafiMoy.potg", "etafiNY.vScore", "etafiVar.eps", "etafiNY.roic",  "etafiVar.roce", "etafiNY.implGeva", "etafiVar.eva", "etafiVar.capex", "etafiVar.roeci"]
//
var armet2 = ["etafiNY.vScore", "etafiVar.eps", "etafiVar.roeci", "n3Score", "n1Score"]
//"n2Score", "etafiMoy.fcfy", 
func meta()->() {
    var arres = [Result]()
    for ivar in armet {
        let n = NN63kg(ni:6, nh:3, no:1)
//        n.determineAddCoefs()
        n.metavar = ivar
        n.batchtrain(7000)
        print("\(n.metavar) *** ddl: \(n.ddl()), pop: \(n.npop), ratiotrain: \(n.ratiotrain()), corr: \(n.corr)\n     pbr modele/reel: moy:\(n.moy), median:\(n.median)")
//        print("\(n.metavar) *** ddl: \(n.ddl()), pop: \(n.npop), ratiotrain: \(n.ratiotrain()), corr: \(n.corr), cluster:\(n.pop)\n     pbr modele/reel: moy:\(n.moy), median:\(n.median)") // pour NNd
        var ires = Result(noml: n.nom)
        ires.mutvar = n.metavar
        ires.r2 = Float(n.r2)
        ires.corr = Float(n.corr)
        ires.ddl = UInt32(n.ddl())
        ires.npop = UInt(n.npop)
        ires.moy = Float(n.moy)
        ires.median = Float(n.median)
        arres.append(ires)
    }
    let ar01 = arres.sort {$0.r2 >= $1.r2}
    for i in 0...3 {
        let f1 = ar01[i]
        print("***\n\(f1.nom)-\(f1.mutvar) *** r2: \(f1.r2) ** ddl: \(f1.ddl), pop: \(f1.npop), ratiotrain: \(f1.tr()), corr: \(f1.corr)\n     pbr modele/reel: moy:\(f1.moy), median:\(f1.median)")
    }
}
train()
