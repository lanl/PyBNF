<?xml version="1.0" encoding="UTF-8"?>
<!-- generated with COPASI 4.22 (Build 170) (http://www.copasi.org) at 2018-04-10 21:50:54 UTC -->
<?oxygen RNGSchema="http://www.copasi.org/static/schema/CopasiML.rng" type="xml"?>
<COPASI xmlns="http://www.copasi.org/static/schema" versionMajor="4" versionMinor="22" versionDevel="170" copasiSourcesModified="0">
  <ListOfFunctions>
    <Function key="Function_6" name="Constant flux (irreversible)" type="PreDefined" reversible="false">
      <Expression>
        v
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_49" name="v" order="0" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_13" name="Mass action (irreversible)" type="MassAction" reversible="false">
      <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <rdf:Description rdf:about="#Function_13">
   <CopasiMT:is rdf:resource="urn:miriam:obo.sbo:SBO:0000041" />
   </rdf:Description>
   </rdf:RDF>
      </MiriamAnnotation>
      <Comment>
        <body xmlns="http://www.w3.org/1999/xhtml">
<b>Mass action rate law for first order irreversible reactions</b>
<p>
Reaction scheme where the products are created from the reactants and the change of a product quantity is proportional to the product of reactant activities. The reaction scheme does not include any reverse process that creates the reactants from the products. The change of a product quantity is proportional to the quantity of one reactant.
</p>
</body>
      </Comment>
      <Expression>
        k1*PRODUCT&lt;substrate_i>
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_81" name="k1" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_79" name="substrate" order="1" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_40" name="Heav" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_40">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:04:05Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        if(x lt 0,0,1)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_264" name="x" order="0" role="variable"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_41" name="Sigmoid" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_41">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:05:38Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        total/(1+exp(-sigma)^wfunction)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_262" name="total" order="0" role="variable"/>
        <ParameterDescription key="FunctionParameter_246" name="sigma" order="1" role="variable"/>
        <ParameterDescription key="FunctionParameter_265" name="wfunction" order="2" role="variable"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_42" name="Rate Law for Growth_1" type="UserDefined" reversible="false">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_42">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:10:38Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        mu*V
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_254" name="V" order="0" role="product"/>
        <ParameterDescription key="FunctionParameter_266" name="mu" order="1" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_43" name="Rate Law for Cln3 Synth_1" type="UserDefined" reversible="false">
      <Expression>
        ks_n3*Dn3*V/(Jn3+Dn3*V)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_268" name="Dn3" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_272" name="Jn3" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_270" name="V" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_267" name="ks_n3" order="3" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_44" name="Rate Law for Bck2 Synth_1" type="UserDefined" reversible="false">
      <Expression>
        V*ks_k2
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_273" name="V" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_269" name="ks_k2" order="1" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_45" name="Rate Law for WHI5deP Synth_1" type="UserDefined" reversible="true">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_45">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:10:40Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        gamma*(Sigmoid(WHI5T,sig,kdp_i5+kdp_i5_14*CDC14-kp_i5-kp_i5_n3*CLN3-kp_i5_k2*BCK2-kp_i5_n2*CLN2-kp_i5_b5*CLB5)-WHI5deP)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_292" name="BCK2" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_282" name="CDC14" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_300" name="CLB5" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_296" name="CLN2" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_288" name="CLN3" order="4" role="modifier"/>
        <ParameterDescription key="FunctionParameter_274" name="WHI5T" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_302" name="WHI5deP" order="6" role="product"/>
        <ParameterDescription key="FunctionParameter_258" name="gamma" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_278" name="kdp_i5" order="8" role="constant"/>
        <ParameterDescription key="FunctionParameter_280" name="kdp_i5_14" order="9" role="constant"/>
        <ParameterDescription key="FunctionParameter_284" name="kp_i5" order="10" role="constant"/>
        <ParameterDescription key="FunctionParameter_298" name="kp_i5_b5" order="11" role="constant"/>
        <ParameterDescription key="FunctionParameter_290" name="kp_i5_k2" order="12" role="constant"/>
        <ParameterDescription key="FunctionParameter_294" name="kp_i5_n2" order="13" role="constant"/>
        <ParameterDescription key="FunctionParameter_286" name="kp_i5_n3" order="14" role="constant"/>
        <ParameterDescription key="FunctionParameter_276" name="sig" order="15" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_46" name="Rate Law for SBFdeP Synth_1" type="UserDefined" reversible="true">
      <Expression>
        gamma*(Sigmoid(SBFT,sig,kdp_bf-kp_bf_b2*CLB2)-SBFdeP)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_283" name="CLB2" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_299" name="SBFT" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_279" name="SBFdeP" order="2" role="product"/>
        <ParameterDescription key="FunctionParameter_303" name="gamma" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_291" name="kdp_bf" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_287" name="kp_bf_b2" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_295" name="sig" order="6" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_47" name="Rate Law for Cln2 Synth_1" type="UserDefined" reversible="false">
      <Expression>
        ks_n2+ks_n2_bf*SBF
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_293" name="SBF" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_277" name="ks_n2" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_285" name="ks_n2_bf" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_48" name="Rate Law for CKIT Synth_1" type="UserDefined" reversible="false">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_48">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:10:36Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        ks_ki+ks_ki_swi5*SWI5A
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_275" name="SWI5A" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_297" name="ks_ki" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_281" name="ks_ki_swi5" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_49" name="Rate Law for CKIT Degr_1" type="UserDefined" reversible="false">
      <Expression>
        kd_ki*(CKIT-CKIP)+kd_kip*CKIP
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_305" name="CKIP" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_289" name="CKIT" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_271" name="kd_ki" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_307" name="kd_kip" order="3" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_50" name="Rate Law for CKIP Synth_1" type="UserDefined" reversible="true">
      <Expression>
        gammaki*(Sigmoid(CKIT,sig,kp_ki_e*(e_ki_n3*CLN3+e_ki_k2*BCK2+e_ki_n2*CLN2+e_ki_b5*CLB5+e_ki_b2*CLB2)-kdp_ki-kdp_ki_14*CDC14)-CKIP)-kd_kip*CKIP
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_319" name="BCK2" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_337" name="CDC14" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_339" name="CKIP" order="2" role="product"/>
        <ParameterDescription key="FunctionParameter_304" name="CKIT" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_331" name="CLB2" order="4" role="modifier"/>
        <ParameterDescription key="FunctionParameter_327" name="CLB5" order="5" role="modifier"/>
        <ParameterDescription key="FunctionParameter_323" name="CLN2" order="6" role="modifier"/>
        <ParameterDescription key="FunctionParameter_315" name="CLN3" order="7" role="modifier"/>
        <ParameterDescription key="FunctionParameter_329" name="e_ki_b2" order="8" role="constant"/>
        <ParameterDescription key="FunctionParameter_325" name="e_ki_b5" order="9" role="constant"/>
        <ParameterDescription key="FunctionParameter_317" name="e_ki_k2" order="10" role="constant"/>
        <ParameterDescription key="FunctionParameter_321" name="e_ki_n2" order="11" role="constant"/>
        <ParameterDescription key="FunctionParameter_313" name="e_ki_n3" order="12" role="constant"/>
        <ParameterDescription key="FunctionParameter_308" name="gammaki" order="13" role="constant"/>
        <ParameterDescription key="FunctionParameter_341" name="kd_kip" order="14" role="constant"/>
        <ParameterDescription key="FunctionParameter_333" name="kdp_ki" order="15" role="constant"/>
        <ParameterDescription key="FunctionParameter_335" name="kdp_ki_14" order="16" role="constant"/>
        <ParameterDescription key="FunctionParameter_311" name="kp_ki_e" order="17" role="constant"/>
        <ParameterDescription key="FunctionParameter_309" name="sig" order="18" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_51" name="Rate Law for Clb5T Synth_1" type="UserDefined" reversible="false">
      <Expression>
        ks_b5+ks_b5_bf*SBF
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_334" name="SBF" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_342" name="ks_b5" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_338" name="ks_b5_bf" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_52" name="Rate Law for Clb5T Degr_1" type="UserDefined" reversible="false">
      <Expression>
        (kd_b5+kd_b5_20*CDC20A+kd_b5_20_i*CDC20A_APC)*CLB5T
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_328" name="CDC20A" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_320" name="CDC20A_APC" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_316" name="CLB5T" order="2" role="substrate"/>
        <ParameterDescription key="FunctionParameter_332" name="kd_b5" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_340" name="kd_b5_20" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_324" name="kd_b5_20_i" order="5" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_53" name="Rate Law for Clb2T Synth_1" type="UserDefined" reversible="false">
      <Expression>
        (ks_b2+ks_b2_m1*MCM1)*V
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_330" name="MCM1" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_312" name="V" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_314" name="ks_b2" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_322" name="ks_b2_m1" order="3" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_54" name="Rate Law for Clb2T Degr_1" type="UserDefined" reversible="false">
      <Expression>
        (kd_b2+kd_b2_20*CDC20A+kd_b2_20_i*CDC20A_APC+kd_b2_h1*CDH1A)*CLB2T
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_301" name="CDC20A" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_345" name="CDC20A_APC" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_349" name="CDH1A" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_351" name="CLB2T" order="3" role="substrate"/>
        <ParameterDescription key="FunctionParameter_310" name="kd_b2" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_326" name="kd_b2_20" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_343" name="kd_b2_20_i" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_347" name="kd_b2_h1" order="7" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_55" name="Rate Law for BUD Synth_1" type="UserDefined" reversible="false">
      <Expression>
        ks_bud_e*(e_bud_n3*CLN3+e_bud_n2*CLN2+e_bud_b5*CLB5+e_bud_b2*CLB2)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_361" name="CLB2" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_357" name="CLB5" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_353" name="CLN2" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_344" name="CLN3" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_359" name="e_bud_b2" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_355" name="e_bud_b5" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_318" name="e_bud_n2" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_348" name="e_bud_n3" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_352" name="ks_bud_e" order="8" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_56" name="Rate Law for ORI Synth_1" type="UserDefined" reversible="false">
      <Expression>
        ks_ori_e*(e_ori_b5*CLB5+e_ori_b2*CLB2)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_350" name="CLB2" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_354" name="CLB5" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_306" name="e_ori_b2" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_358" name="e_ori_b5" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_362" name="ks_ori_e" order="4" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_57" name="Rate Law for SPN Synth_1" type="UserDefined" reversible="false">
      <Expression>
        ks_spn*Heav(CLB2-Jspn)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_336" name="CLB2" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_360" name="Jspn" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_363" name="ks_spn" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_58" name="Rate Law for SWI5T Synth_1" type="UserDefined" reversible="false">
      <Expression>
        ks_swi5+ks_swi5_m1*MCM1
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_366" name="MCM1" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_364" name="ks_swi5" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_346" name="ks_swi5_m1" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_59" name="Rate Law for CDC20T Synth_1" type="UserDefined" reversible="false">
      <Expression>
        ks_20+ks_20_m1*MCM1
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_369" name="MCM1" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_367" name="ks_20" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_356" name="ks_20_m1" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_60" name="Rate Law for CDC20A_APCP Synth_1" type="UserDefined" reversible="true">
      <Expression>
        gamma*(Sigmoid(CDC20A_APCP_T,sig,ka_20-ki_20_ori*Heav(ORI-1)*(1-Heav(SPN-1)))-CDC20A)-kd_20*CDC20A
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_382" name="CDC20A" order="0" role="product"/>
        <ParameterDescription key="FunctionParameter_365" name="CDC20A_APCP_T" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_378" name="ORI" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_380" name="SPN" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_370" name="gamma" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_374" name="ka_20" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_384" name="kd_20" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_376" name="ki_20_ori" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_372" name="sig" order="8" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_61" name="Rate Law for APCP Synth_1" type="UserDefined" reversible="true">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_61">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:10:38Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        gammacp*(Sigmoid(APCPT,sig,ka_cp_b2*CLB2-ki_cp)-APCP)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_389" name="APCP" order="0" role="product"/>
        <ParameterDescription key="FunctionParameter_381" name="APCPT" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_368" name="CLB2" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_385" name="gammacp" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_373" name="ka_cp_b2" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_387" name="ki_cp" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_377" name="sig" order="6" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_62" name="Rate Law for CDH1A Synth_1" type="UserDefined" reversible="true">
      <Expression>
        gamma*(Sigmoid(CDH1T,sig,ka_h1+ka_h1_14*CDC14-ki_h1-ki_h1_e*(e_h1_n3*CLN3+e_h1_n2*CLN2+e_h1_b5*CLB5+e_h1_b2*CLB2))-CDH1A)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_394" name="CDC14" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_416" name="CDH1A" order="1" role="product"/>
        <ParameterDescription key="FunctionParameter_386" name="CDH1T" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_414" name="CLB2" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_410" name="CLB5" order="4" role="modifier"/>
        <ParameterDescription key="FunctionParameter_406" name="CLN2" order="5" role="modifier"/>
        <ParameterDescription key="FunctionParameter_402" name="CLN3" order="6" role="modifier"/>
        <ParameterDescription key="FunctionParameter_412" name="e_h1_b2" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_408" name="e_h1_b5" order="8" role="constant"/>
        <ParameterDescription key="FunctionParameter_404" name="e_h1_n2" order="9" role="constant"/>
        <ParameterDescription key="FunctionParameter_400" name="e_h1_n3" order="10" role="constant"/>
        <ParameterDescription key="FunctionParameter_390" name="gamma" order="11" role="constant"/>
        <ParameterDescription key="FunctionParameter_383" name="ka_h1" order="12" role="constant"/>
        <ParameterDescription key="FunctionParameter_392" name="ka_h1_14" order="13" role="constant"/>
        <ParameterDescription key="FunctionParameter_396" name="ki_h1" order="14" role="constant"/>
        <ParameterDescription key="FunctionParameter_398" name="ki_h1_e" order="15" role="constant"/>
        <ParameterDescription key="FunctionParameter_375" name="sig" order="16" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_63" name="Rate Law for NET1deP Synth_1" type="UserDefined" reversible="true">
      <Expression>
        gamma*(Sigmoid(NET1T,signet,kdp_net+kdp_net_14*CDC14+kdp_net_px*PPX-kp_net-kp_net_b2*CLB2-kp_net_en*MEN-kp_net_15*CDC15)-NET1deP)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_397" name="CDC14" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_429" name="CDC15" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_421" name="CLB2" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_425" name="MEN" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_413" name="NET1T" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_431" name="NET1deP" order="5" role="product"/>
        <ParameterDescription key="FunctionParameter_379" name="PPX" order="6" role="modifier"/>
        <ParameterDescription key="FunctionParameter_417" name="gamma" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_405" name="kdp_net" order="8" role="constant"/>
        <ParameterDescription key="FunctionParameter_401" name="kdp_net_14" order="9" role="constant"/>
        <ParameterDescription key="FunctionParameter_393" name="kdp_net_px" order="10" role="constant"/>
        <ParameterDescription key="FunctionParameter_388" name="kp_net" order="11" role="constant"/>
        <ParameterDescription key="FunctionParameter_427" name="kp_net_15" order="12" role="constant"/>
        <ParameterDescription key="FunctionParameter_419" name="kp_net_b2" order="13" role="constant"/>
        <ParameterDescription key="FunctionParameter_423" name="kp_net_en" order="14" role="constant"/>
        <ParameterDescription key="FunctionParameter_409" name="signet" order="15" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_64" name="Rate Law for PPX Synth_1" type="UserDefined" reversible="true">
      <Expression>
        gamma*(Sigmoid(PPXT,sig,ka_px-ki_px-ki_px_p1*ESP1)-PPX)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_403" name="ESP1" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_411" name="PPX" order="1" role="product"/>
        <ParameterDescription key="FunctionParameter_428" name="PPXT" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_432" name="gamma" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_420" name="ka_px" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_371" name="ki_px" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_395" name="ki_px_p1" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_424" name="sig" order="7" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_65" name="Rate Law for PDS1T Degr_1" type="UserDefined" reversible="false">
      <Expression>
        (kd_pds+ks_pds_20*CDC20A+kd_pds_20_i*CDC20A_APC)*PDS1T
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_418" name="CDC20A" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_433" name="CDC20A_APC" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_435" name="PDS1T" order="2" role="substrate"/>
        <ParameterDescription key="FunctionParameter_415" name="kd_pds" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_426" name="kd_pds_20_i" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_399" name="ks_pds_20" order="5" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_66" name="Rate Law for CDC15 Synth_1" type="UserDefined" reversible="true">
      <Expression>
        gamma*(Sigmoid(CDC15T,sig,ka_15+ka_15_14*CDC14-ki_15-ki_15_b2*CLB2)-CDC15)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_441" name="CDC14" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_449" name="CDC15" order="1" role="product"/>
        <ParameterDescription key="FunctionParameter_430" name="CDC15T" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_447" name="CLB2" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_436" name="gamma" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_437" name="ka_15" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_439" name="ka_15_14" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_443" name="ki_15" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_445" name="ki_15_b2" order="8" role="constant"/>
        <ParameterDescription key="FunctionParameter_391" name="sig" order="9" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_67" name="Rate Law for TEM1 Synth_1" type="UserDefined" reversible="true">
      <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_67">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:10:39Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
      </MiriamAnnotation>
      <Expression>
        gammatem*(Sigmoid(TEM1T,sig,ka_tem+ka_tem_lo*POLOA+ka_tem_p1*ESP1-ki_tem-ki_tem_px*PPX)-TEM1)
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_455" name="ESP1" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_451" name="POLOA" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_461" name="PPX" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_463" name="TEM1" order="3" role="product"/>
        <ParameterDescription key="FunctionParameter_446" name="TEM1T" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_450" name="gammatem" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_438" name="ka_tem" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_422" name="ka_tem_lo" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_453" name="ka_tem_p1" order="8" role="constant"/>
        <ParameterDescription key="FunctionParameter_457" name="ki_tem" order="9" role="constant"/>
        <ParameterDescription key="FunctionParameter_459" name="ki_tem_px" order="10" role="constant"/>
        <ParameterDescription key="FunctionParameter_442" name="sig" order="11" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_68" name="Rate Law for POLOT Synth_1" type="UserDefined" reversible="false">
      <Expression>
        ks_lo+ks_lo_m1*MCM1
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_456" name="MCM1" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_464" name="ks_lo" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_460" name="ks_lo_m1" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_69" name="Rate Law for POLOT Degr_1" type="UserDefined" reversible="false">
      <Expression>
        (kd_lo+kd_lo_h1*CDH1A)*POLOT
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_434" name="CDH1A" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_440" name="POLOT" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_454" name="kd_lo" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_462" name="kd_lo_h1" order="3" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_70" name="Rate Law for POLOA Synth_1" type="UserDefined" reversible="true">
      <Expression>
        gamma*(Sigmoid(POLOT,sig,ka_lo+ka_lo_b2*CLB2-ki_lo)-POLOA)-(kd_lo+kd_lo_h1*CDH1A)*POLOA
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_480" name="CDH1A" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_470" name="CLB2" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_474" name="POLOA" order="2" role="product"/>
        <ParameterDescription key="FunctionParameter_452" name="POLOT" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_444" name="gamma" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_466" name="ka_lo" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_468" name="ka_lo_b2" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_476" name="kd_lo" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_478" name="kd_lo_h1" order="8" role="constant"/>
        <ParameterDescription key="FunctionParameter_472" name="ki_lo" order="9" role="constant"/>
        <ParameterDescription key="FunctionParameter_448" name="sig" order="10" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_71" name="Rate Law for CDC20A_APC Synth_1" type="UserDefined" reversible="true">
      <Expression>
        gamma*(Sigmoid(CDC20A_APC_T,sig,ka_20-ki_20_ori*Heav(ORI-1)*(1-Heav(SPN-1)))-CDC20A_APC)-kd_20*CDC20A_APC
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_485" name="CDC20A_APC" order="0" role="product"/>
        <ParameterDescription key="FunctionParameter_477" name="CDC20A_APC_T" order="1" role="modifier"/>
        <ParameterDescription key="FunctionParameter_407" name="ORI" order="2" role="modifier"/>
        <ParameterDescription key="FunctionParameter_483" name="SPN" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_481" name="gamma" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_469" name="ka_20" order="5" role="constant"/>
        <ParameterDescription key="FunctionParameter_487" name="kd_20" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_465" name="ki_20_ori" order="7" role="constant"/>
        <ParameterDescription key="FunctionParameter_473" name="sig" order="8" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
  </ListOfFunctions>
  <Model key="Model_1" name="Yeast Cell Cycle" simulationType="time" timeUnit="min" volumeUnit="l" areaUnit="m²" lengthUnit="m" quantityUnit="mol" type="deterministic" avogadroConstant="6.0221408570000002e+23">
    <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:vCard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <rdf:Description rdf:about="#Model_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T09:04:54Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <dcterms:creator>
      <rdf:Bag>
        <rdf:li>
          <rdf:Description>
            <vCard:N>
              <rdf:Description>
                <vCard:Family>Mitra</vCard:Family>
                <vCard:Given>Eshan</vCard:Given>
              </rdf:Description>
            </vCard:N>
            <vCard:ORG>
              <rdf:Description>
                <vCard:Orgname>Los Alamos National Lab</vCard:Orgname>
              </rdf:Description>
            </vCard:ORG>
          </rdf:Description>
        </rdf:li>
      </rdf:Bag>
    </dcterms:creator>
    <dcterms:modified>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T09:04:54Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:modified>
  </rdf:Description>
</rdf:RDF>

    </MiriamAnnotation>
    <Comment>
      <body xmlns="http://www.w3.org/1999/xhtml">
    <pre>Model of the yeast cell cycle originally described in Oguz et al (2013) "Optimization and model reduction in the high dimensional parameter space of a budding yeast cell cycle model" BMC Syst. Biol.
Adapted to SBML format and used as an example problem in Mitra et al (2018) "Using both qualitative and quantitative data to imporve parameter identification for systems biology models".</pre>
  </body>
    </Comment>
    <ListOfCompartments>
      <Compartment key="Compartment_1" name="cell" simulationType="fixed" dimensionality="3">
      </Compartment>
    </ListOfCompartments>
    <ListOfMetabolites>
      <Metabolite key="Metabolite_68" name="V" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_68">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T09:10:10Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_69" name="BCK2" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_69">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T09:13:02Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_70" name="CLN3" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_70">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:04:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_71" name="WHI5deP" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_71">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:53Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_72" name="SBFdeP" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_72">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_73" name="CLN2" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_73">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:18:34Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_74" name="CKIT" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_74">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_75" name="CKIP" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_75">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_76" name="CLB5T" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_76">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:17:21Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_77" name="CLB2T" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_77">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_78" name="BUD" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_78">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_79" name="ORI" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_79">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_80" name="SPN" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_80">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:53Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_81" name="SWI5T" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_81">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:17:13Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_82" name="CDC20T" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_82">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_83" name="CDC20A_APCP" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_83">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_84" name="APCP" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_84">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_85" name="CDH1A" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_85">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:18:34Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_86" name="NET1deP" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_86">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:17:18Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_87" name="PPX" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_87">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_88" name="PDS1T" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_88">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:53Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_89" name="CDC15" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_89">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:17:10Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_90" name="TEM1" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_90">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:26:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_91" name="POLOT" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_91">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:35:48Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_92" name="POLOA" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_93" name="CDC20A_APC" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_93">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T16:21:22Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_94" name="FuncSafety" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_94">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:04:57Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          Heav(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[APCP],Reference=Concentration>)+Sigmoid(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[APCPT],Reference=Value>,1,1)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_95" name="CLB5" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_95">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:19:30Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5T],Reference=Concentration>*(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5T],Reference=Concentration>+&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2T],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CKIT],Reference=Concentration>)/(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5T],Reference=Concentration>+&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2T],Reference=Concentration>+0.001)*Heav(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5T],Reference=Concentration>+&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2T],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CKIT],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_96" name="CLB2" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_96">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:22:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2T],Reference=Concentration>*(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5T],Reference=Concentration>+&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2T],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CKIT],Reference=Concentration>)/(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5T],Reference=Concentration>+&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2T],Reference=Concentration>+0.001)*Heav(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5T],Reference=Concentration>+&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2T],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CKIT],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_97" name="SBF" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_97">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:24:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          (&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SBFdeP],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[WHI5deP],Reference=Concentration>)*Heav(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SBFdeP],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[WHI5deP],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_98" name="CDC14" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_98">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:27:43Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          (&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[CDC14T],Reference=Value>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kas_net],Reference=Value>*&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[NET1deP],Reference=Concentration>)*Heav(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[CDC14T],Reference=Value>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kas_net],Reference=Value>*&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[NET1deP],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_99" name="ESP1" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_99">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:29:17Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          (&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ESP1T],Reference=Value>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[PDS1T],Reference=Concentration>)*Heav(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ESP1T],Reference=Value>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[PDS1T],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_100" name="MEN" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_100">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:31:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[TEM1],Reference=Concentration> lt &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC15],Reference=Concentration>,&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[TEM1],Reference=Concentration>,&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC15],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_101" name="MCM1" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_101">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:32:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          Sigmoid(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[MCM1T],Reference=Value>,&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=Value>,&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_m1_b2],Reference=Value>*&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_m1],Reference=Value>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_102" name="SWI5A" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_102">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T12:36:03Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          Sigmoid(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SWI5T],Reference=Concentration>,&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=Value>,&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_swi5_14],Reference=Value>*&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC14],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_swi5_b2],Reference=Value>*&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_103" name="CDC20A_APCP_T" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_103">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:15:07Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20T],Reference=Concentration> lt &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[APCP],Reference=Concentration>,&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20T],Reference=Concentration>,&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[APCP],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_104" name="CDC20A_APC_T" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_104">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:46:04Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          if(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20T],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20A_APCP_T],Reference=Concentration> lt &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[APCPT],Reference=Value>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[APCP],Reference=Concentration>,&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20T],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20A_APCP_T],Reference=Concentration>,&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[APCPT],Reference=Value>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[APCP],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_105" name="DIV_COUNT" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_106" name="FLAG_BUD" simulationType="reactions" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_106">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T16:21:20Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Metabolite>
      <Metabolite key="Metabolite_107" name="FLAG_UDNA" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_108" name="FLAG_SPC" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_109" name="dCKIT_sign" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_110" name="CKIT_peaks" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_111" name="OffsetTime" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_111">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T16:15:22Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Reference=Time>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[phi_15],Reference=Value>
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_112" name="VDiv1" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_113" name="VDiv2" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_114" name="VDiv3" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_115" name="Diff1" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_115">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T16:35:20Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          abs(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[VDiv2],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[VDiv1],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_116" name="Diff2" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_116">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T16:36:09Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          abs(&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[VDiv3],Reference=Concentration>-&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[VDiv2],Reference=Concentration>)
        </Expression>
      </Metabolite>
      <Metabolite key="Metabolite_117" name="dCln3_sign" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_118" name="dBck2_sign" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_119" name="dCln2_sign" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_120" name="dClb5T_sign" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_121" name="dClb2T_sign" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_122" name="dSWI5T_sign" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_123" name="dCDC20T_sign" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_124" name="dPDS1T_sign" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_125" name="dPOLOT_sign" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_126" name="Cln3_peaks" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_127" name="Bck2_peaks" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_128" name="Cln2_peaks" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_129" name="Clb5T_peaks" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_130" name="Clb2T_peaks" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_131" name="SWI5T_peaks" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_132" name="CDC20T_peaks" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_133" name="PDS1T_peaks" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_134" name="POLOT_peaks" simulationType="reactions" compartment="Compartment_1">
      </Metabolite>
      <Metabolite key="Metabolite_135" name="CLB2CLB5" simulationType="assignment" compartment="Compartment_1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Metabolite_135">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:22:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2],Reference=Concentration>+&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5],Reference=Concentration>
        </Expression>
      </Metabolite>
    </ListOfMetabolites>
    <ListOfModelValues>
      <ModelValue key="ModelValue_128" name="mdt" simulationType="fixed">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_128">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T09:20:41Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_129" name="mu" simulationType="assignment">
        <MiriamAnnotation>
<rdf:RDF xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_129">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T09:19:44Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:unknown rdf:resource="#" />
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <Expression>
          log(2)/&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[mdt],Reference=Value>
        </Expression>
      </ModelValue>
      <ModelValue key="ModelValue_130" name="ks_n3" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_131" name="Dn3" simulationType="fixed">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_131">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T09:59:45Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_132" name="Jn3" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_133" name="kd_n3" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_134" name="gamma" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_135" name="gammaki" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_136" name="gammacp" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_137" name="gammatem" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_138" name="sig" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_139" name="signet" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_140" name="ks_k2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_141" name="kd_k2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_142" name="kdp_i5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_143" name="kdp_i5_14" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_144" name="kp_i5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_145" name="kp_i5_n3" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_146" name="kp_i5_k2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_147" name="kp_i5_n2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_148" name="kp_i5_b5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_149" name="kdp_bf" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_150" name="kp_bf_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_151" name="ks_n2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_152" name="ks_n2_bf" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_153" name="kd_n2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_154" name="ks_ki" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_155" name="ks_ki_swi5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_156" name="kd_ki" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_157" name="kd_kip" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_158" name="kp_ki_e" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_159" name="e_ki_n3" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_160" name="e_ki_k2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_161" name="e_ki_n2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_162" name="e_ki_b5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_163" name="e_ki_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_164" name="kdp_ki" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_165" name="kdp_ki_14" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_166" name="ks_b5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_167" name="ks_b5_bf" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_168" name="kd_b5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_169" name="kd_b5_20" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_170" name="ks_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_171" name="ks_b2_m1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_172" name="kd_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_173" name="kd_b2_20" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_174" name="kd_b2_h1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_175" name="ks_bud_e" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_176" name="e_bud_n3" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_177" name="e_bud_n2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_178" name="e_bud_b5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_179" name="e_bud_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_180" name="kd_bud" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_181" name="ks_spn" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_182" name="kd_spn" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_183" name="Jspn" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_184" name="ks_ori_e" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_185" name="e_ori_b5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_186" name="e_ori_b2" simulationType="fixed">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_186">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:30:39Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_187" name="kd_ori" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_188" name="ks_swi5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_189" name="ks_swi5_m1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_190" name="kd_swi5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_191" name="ka_swi5_14" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_192" name="ki_swi5_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_193" name="ka_m1_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_194" name="ki_m1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_195" name="ks_20" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_196" name="ks_20_m1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_197" name="kd_20" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_198" name="ka_20" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_199" name="kd_b5_20_i" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_200" name="kd_b2_20_i" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_201" name="ki_20_ori" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_202" name="ka_cp_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_203" name="ki_cp" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_204" name="ka_h1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_205" name="ka_h1_14" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_206" name="ki_h1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_207" name="ki_h1_e" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_208" name="e_h1_n3" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_209" name="e_h1_n2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_210" name="e_h1_b5" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_211" name="e_h1_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_212" name="kdp_net" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_213" name="kdp_net_14" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_214" name="kdp_net_px" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_215" name="kp_net" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_216" name="kp_net_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_217" name="kp_net_en" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_218" name="kp_net_15" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_219" name="ka_px" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_220" name="ki_px" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_221" name="ki_px_p1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_222" name="ks_pds" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_223" name="kd_pds" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_224" name="kd_pds_20" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_225" name="ka_15" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_226" name="ka_15_14" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_227" name="ki_15" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_228" name="ki_15_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_229" name="ka_tem" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_230" name="ka_tem_lo" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_231" name="ka_tem_p1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_232" name="ki_tem" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_233" name="ki_tem_px" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_234" name="ks_lo" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_235" name="ks_lo_m1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_236" name="kd_lo" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_237" name="kd_lo_h1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_238" name="ka_lo" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_239" name="ka_lo_b2" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_240" name="ki_lo" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_241" name="kas_net" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_242" name="WHI5T" simulationType="fixed">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#ModelValue_242">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T11:01:00Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </ModelValue>
      <ModelValue key="ModelValue_243" name="SBFT" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_244" name="MCM1T" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_245" name="APCPT" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_246" name="CDH1T" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_247" name="NET1T" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_248" name="CDC14T" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_249" name="PPXT" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_250" name="ESP1T" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_251" name="CDC15T" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_252" name="TEM1T" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_253" name="kd_pds_20_i" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_254" name="f" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_255" name="phi_15" simulationType="fixed">
      </ModelValue>
    </ListOfModelValues>
    <ListOfReactions>
      <Reaction key="Reaction_39" name="Growth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_39">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T09:14:28Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_68" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_4521" name="mu" value="0.0099021"/>
        </ListOfConstants>
        <KineticLaw function="Function_42" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_254">
              <SourceParameter reference="Metabolite_68"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_266">
              <SourceParameter reference="ModelValue_129"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_40" name="Cln3 Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_40">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:01:18Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_70" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_68" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4522" name="Dn3" value="0.732"/>
          <Constant key="Parameter_4523" name="Jn3" value="4.27"/>
          <Constant key="Parameter_4524" name="ks_n3" value="1.11"/>
        </ListOfConstants>
        <KineticLaw function="Function_43" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_268">
              <SourceParameter reference="ModelValue_131"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_272">
              <SourceParameter reference="ModelValue_132"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_270">
              <SourceParameter reference="Metabolite_68"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_267">
              <SourceParameter reference="ModelValue_130"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_41" name="Cln3 Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_41">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T10:10:33Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_70" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4525" name="k1" value="0.794"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="ModelValue_133"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_79">
              <SourceParameter reference="Metabolite_70"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_42" name="Bck2 Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_42">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T11:00:07Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_69" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_68" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4526" name="ks_k2" value="0.0553"/>
        </ListOfConstants>
        <KineticLaw function="Function_44" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_273">
              <SourceParameter reference="Metabolite_68"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_269">
              <SourceParameter reference="ModelValue_140"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_43" name="Bck2 Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_43">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T11:02:40Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_69" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4527" name="k1" value="3.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="ModelValue_141"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_79">
              <SourceParameter reference="Metabolite_69"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_44" name="WHI5deP Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_44">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T11:11:35Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_71" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_98" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_70" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_69" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_73" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_95" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4528" name="WHI5T" value="2.1"/>
          <Constant key="Parameter_4529" name="gamma" value="2.22"/>
          <Constant key="Parameter_4530" name="kdp_i5" value="1.22"/>
          <Constant key="Parameter_4531" name="kdp_i5_14" value="0.195"/>
          <Constant key="Parameter_4532" name="kp_i5" value="0.0275"/>
          <Constant key="Parameter_4825" name="kp_i5_b5" value="0.0422"/>
          <Constant key="Parameter_4857" name="kp_i5_k2" value="23.7"/>
          <Constant key="Parameter_4858" name="kp_i5_n2" value="2.97"/>
          <Constant key="Parameter_4859" name="kp_i5_n3" value="6.1"/>
          <Constant key="Parameter_4828" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_45" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_292">
              <SourceParameter reference="Metabolite_69"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_282">
              <SourceParameter reference="Metabolite_98"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_300">
              <SourceParameter reference="Metabolite_95"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_296">
              <SourceParameter reference="Metabolite_73"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_288">
              <SourceParameter reference="Metabolite_70"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_274">
              <SourceParameter reference="ModelValue_242"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_302">
              <SourceParameter reference="Metabolite_71"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_258">
              <SourceParameter reference="ModelValue_134"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_278">
              <SourceParameter reference="ModelValue_142"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_280">
              <SourceParameter reference="ModelValue_143"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_284">
              <SourceParameter reference="ModelValue_144"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_298">
              <SourceParameter reference="ModelValue_148"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_290">
              <SourceParameter reference="ModelValue_146"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_294">
              <SourceParameter reference="ModelValue_147"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_286">
              <SourceParameter reference="ModelValue_145"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_276">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_45" name="SBFdeP Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_45">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T13:24:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_72" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_96" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4829" name="SBFT" value="0.468"/>
          <Constant key="Parameter_4830" name="gamma" value="2.22"/>
          <Constant key="Parameter_4831" name="kdp_bf" value="2.93"/>
          <Constant key="Parameter_4832" name="kp_bf_b2" value="9.36"/>
          <Constant key="Parameter_4833" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_46" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_283">
              <SourceParameter reference="Metabolite_96"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_299">
              <SourceParameter reference="ModelValue_243"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_279">
              <SourceParameter reference="Metabolite_72"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_303">
              <SourceParameter reference="ModelValue_134"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_291">
              <SourceParameter reference="ModelValue_149"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_287">
              <SourceParameter reference="ModelValue_150"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_295">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_46" name="Cln2 Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_46">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T13:31:46Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_73" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_97" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4834" name="ks_n2" value="1e-08"/>
          <Constant key="Parameter_4835" name="ks_n2_bf" value="0.996"/>
        </ListOfConstants>
        <KineticLaw function="Function_47" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_293">
              <SourceParameter reference="Metabolite_97"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_277">
              <SourceParameter reference="ModelValue_151"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_285">
              <SourceParameter reference="ModelValue_152"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_47" name="Cln2 Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_47">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T13:32:56Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_73" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4836" name="k1" value="0.032"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="ModelValue_153"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_79">
              <SourceParameter reference="Metabolite_73"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_48" name="CKIT Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_48">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T13:35:10Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_74" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_102" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4837" name="ks_ki" value="0.00663"/>
          <Constant key="Parameter_4838" name="ks_ki_swi5" value="0.089"/>
        </ListOfConstants>
        <KineticLaw function="Function_48" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_275">
              <SourceParameter reference="Metabolite_102"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_297">
              <SourceParameter reference="ModelValue_154"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_281">
              <SourceParameter reference="ModelValue_155"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_49" name="CKIT Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_49">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T13:42:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_74" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_75" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4839" name="kd_ki" value="0.0524"/>
          <Constant key="Parameter_4840" name="kd_kip" value="0.899"/>
        </ListOfConstants>
        <KineticLaw function="Function_49" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_305">
              <SourceParameter reference="Metabolite_75"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_289">
              <SourceParameter reference="Metabolite_74"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_271">
              <SourceParameter reference="ModelValue_156"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_307">
              <SourceParameter reference="ModelValue_157"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_50" name="CKIP Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_50">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T13:44:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_75" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_74" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_70" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_69" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_73" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_95" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_96" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_98" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4841" name="e_ki_b2" value="3.12"/>
          <Constant key="Parameter_4842" name="e_ki_b5" value="2.39"/>
          <Constant key="Parameter_4843" name="e_ki_k2" value="0.397"/>
          <Constant key="Parameter_4844" name="e_ki_n2" value="19.5"/>
          <Constant key="Parameter_4845" name="e_ki_n3" value="1.05"/>
          <Constant key="Parameter_4846" name="gammaki" value="12.9"/>
          <Constant key="Parameter_4847" name="kd_kip" value="0.899"/>
          <Constant key="Parameter_4848" name="kdp_ki" value="0.836"/>
          <Constant key="Parameter_4849" name="kdp_ki_14" value="1.11"/>
          <Constant key="Parameter_4850" name="kp_ki_e" value="0.65"/>
          <Constant key="Parameter_4851" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_50" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_319">
              <SourceParameter reference="Metabolite_69"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_337">
              <SourceParameter reference="Metabolite_98"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_339">
              <SourceParameter reference="Metabolite_75"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_304">
              <SourceParameter reference="Metabolite_74"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_331">
              <SourceParameter reference="Metabolite_96"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_327">
              <SourceParameter reference="Metabolite_95"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_323">
              <SourceParameter reference="Metabolite_73"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_315">
              <SourceParameter reference="Metabolite_70"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_329">
              <SourceParameter reference="ModelValue_163"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_325">
              <SourceParameter reference="ModelValue_162"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_317">
              <SourceParameter reference="ModelValue_160"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_321">
              <SourceParameter reference="ModelValue_161"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_313">
              <SourceParameter reference="ModelValue_159"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_308">
              <SourceParameter reference="ModelValue_135"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_341">
              <SourceParameter reference="ModelValue_157"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_333">
              <SourceParameter reference="ModelValue_164"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_335">
              <SourceParameter reference="ModelValue_165"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_311">
              <SourceParameter reference="ModelValue_158"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_309">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_51" name="Clb5T Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_51">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T13:50:59Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_76" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_97" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4865" name="ks_b5" value="0.000538"/>
          <Constant key="Parameter_4866" name="ks_b5_bf" value="0.0178"/>
        </ListOfConstants>
        <KineticLaw function="Function_51" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_334">
              <SourceParameter reference="Metabolite_97"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_342">
              <SourceParameter reference="ModelValue_166"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_338">
              <SourceParameter reference="ModelValue_167"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_52" name="Clb5T Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_52">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T13:52:13Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_76" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_83" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_93" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4867" name="kd_b5" value="0.0556"/>
          <Constant key="Parameter_4868" name="kd_b5_20" value="0.0445"/>
          <Constant key="Parameter_4869" name="kd_b5_20_i" value="0.00498"/>
        </ListOfConstants>
        <KineticLaw function="Function_52" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_328">
              <SourceParameter reference="Metabolite_83"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_320">
              <SourceParameter reference="Metabolite_93"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_316">
              <SourceParameter reference="Metabolite_76"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_332">
              <SourceParameter reference="ModelValue_168"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_340">
              <SourceParameter reference="ModelValue_169"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_324">
              <SourceParameter reference="ModelValue_199"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_53" name="Clb2T Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_53">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:00:34Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_77" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_101" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_68" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4870" name="ks_b2" value="0.00762"/>
          <Constant key="Parameter_4871" name="ks_b2_m1" value="0.031"/>
        </ListOfConstants>
        <KineticLaw function="Function_53" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_330">
              <SourceParameter reference="Metabolite_101"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_312">
              <SourceParameter reference="Metabolite_68"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_314">
              <SourceParameter reference="ModelValue_170"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_322">
              <SourceParameter reference="ModelValue_171"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_54" name="Clb2T Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_54">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:01:37Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_77" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_83" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_93" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_85" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4872" name="kd_b2" value="0.00298"/>
          <Constant key="Parameter_4873" name="kd_b2_20" value="0.136"/>
          <Constant key="Parameter_4874" name="kd_b2_20_i" value="0.0374"/>
          <Constant key="Parameter_4875" name="kd_b2_h1" value="0.662"/>
        </ListOfConstants>
        <KineticLaw function="Function_54" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_301">
              <SourceParameter reference="Metabolite_83"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_345">
              <SourceParameter reference="Metabolite_93"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_349">
              <SourceParameter reference="Metabolite_85"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_351">
              <SourceParameter reference="Metabolite_77"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_310">
              <SourceParameter reference="ModelValue_172"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_326">
              <SourceParameter reference="ModelValue_173"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_343">
              <SourceParameter reference="ModelValue_200"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_347">
              <SourceParameter reference="ModelValue_174"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_55" name="BUD Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_55">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:05:26Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_78" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_70" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_73" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_95" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_96" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4876" name="e_bud_b2" value="1.89"/>
          <Constant key="Parameter_4877" name="e_bud_b5" value="3"/>
          <Constant key="Parameter_4878" name="e_bud_n2" value="1.12"/>
          <Constant key="Parameter_4879" name="e_bud_n3" value="0.0078"/>
          <Constant key="Parameter_4880" name="ks_bud_e" value="0.287"/>
        </ListOfConstants>
        <KineticLaw function="Function_55" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_361">
              <SourceParameter reference="Metabolite_96"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_357">
              <SourceParameter reference="Metabolite_95"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_353">
              <SourceParameter reference="Metabolite_73"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_344">
              <SourceParameter reference="Metabolite_70"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_359">
              <SourceParameter reference="ModelValue_179"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_355">
              <SourceParameter reference="ModelValue_178"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_318">
              <SourceParameter reference="ModelValue_177"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_348">
              <SourceParameter reference="ModelValue_176"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_352">
              <SourceParameter reference="ModelValue_175"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_56" name="BUD Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_56">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:06:23Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_78" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4881" name="k1" value="0.059"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="ModelValue_180"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_79">
              <SourceParameter reference="Metabolite_78"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_57" name="ORI Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_57">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:08:50Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_79" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_95" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_96" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4882" name="e_ori_b2" value="0.124"/>
          <Constant key="Parameter_4883" name="e_ori_b5" value="5.04"/>
          <Constant key="Parameter_4884" name="ks_ori_e" value="1.9"/>
        </ListOfConstants>
        <KineticLaw function="Function_56" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_350">
              <SourceParameter reference="Metabolite_96"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_354">
              <SourceParameter reference="Metabolite_95"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_306">
              <SourceParameter reference="ModelValue_186"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_358">
              <SourceParameter reference="ModelValue_185"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_362">
              <SourceParameter reference="ModelValue_184"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_58" name="ORI Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_58">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:09:26Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_79" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4885" name="k1" value="0.0817"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="ModelValue_187"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_79">
              <SourceParameter reference="Metabolite_79"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_59" name="SPN Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_59">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:09:51Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_80" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_96" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4886" name="Jspn" value="0.809"/>
          <Constant key="Parameter_4887" name="ks_spn" value="0.0743"/>
        </ListOfConstants>
        <KineticLaw function="Function_57" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_336">
              <SourceParameter reference="Metabolite_96"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_360">
              <SourceParameter reference="ModelValue_183"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_363">
              <SourceParameter reference="ModelValue_181"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_60" name="SPN Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_60">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:10:24Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_80" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4888" name="k1" value="0.0384"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="ModelValue_182"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_79">
              <SourceParameter reference="Metabolite_80"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_61" name="SWI5T Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_61">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:11:54Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_81" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_101" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4889" name="ks_swi5" value="0.00558"/>
          <Constant key="Parameter_4890" name="ks_swi5_m1" value="0.0389"/>
        </ListOfConstants>
        <KineticLaw function="Function_58" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_366">
              <SourceParameter reference="Metabolite_101"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_364">
              <SourceParameter reference="ModelValue_188"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_346">
              <SourceParameter reference="ModelValue_189"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_62" name="SWI5T Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_62">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:13:43Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_81" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4891" name="k1" value="0.042"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="ModelValue_190"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_79">
              <SourceParameter reference="Metabolite_81"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_63" name="CDC20T Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_63">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:11:31Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_82" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_101" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4892" name="ks_20" value="0.0221"/>
          <Constant key="Parameter_4893" name="ks_20_m1" value="0.354"/>
        </ListOfConstants>
        <KineticLaw function="Function_59" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_369">
              <SourceParameter reference="Metabolite_101"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_367">
              <SourceParameter reference="ModelValue_195"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_356">
              <SourceParameter reference="ModelValue_196"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_64" name="CDC20T Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_64">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:14:33Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_82" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfConstants>
          <Constant key="Parameter_4894" name="k1" value="0.124"/>
        </ListOfConstants>
        <KineticLaw function="Function_13" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_81">
              <SourceParameter reference="ModelValue_197"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_79">
              <SourceParameter reference="Metabolite_82"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_65" name="CDC20A_APCP Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_65">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:17:25Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_83" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_103" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_79" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_80" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4895" name="gamma" value="2.22"/>
          <Constant key="Parameter_4896" name="ka_20" value="0.0104"/>
          <Constant key="Parameter_4897" name="kd_20" value="0.124"/>
          <Constant key="Parameter_4903" name="ki_20_ori" value="5.04"/>
          <Constant key="Parameter_4902" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_60" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_382">
              <SourceParameter reference="Metabolite_83"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_365">
              <SourceParameter reference="Metabolite_103"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_378">
              <SourceParameter reference="Metabolite_79"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_380">
              <SourceParameter reference="Metabolite_80"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_370">
              <SourceParameter reference="ModelValue_134"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_374">
              <SourceParameter reference="ModelValue_198"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_384">
              <SourceParameter reference="ModelValue_197"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_376">
              <SourceParameter reference="ModelValue_201"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_372">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_66" name="APCP Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_66">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:21:13Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_84" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_96" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4901" name="APCPT" value="45.7"/>
          <Constant key="Parameter_4898" name="gammacp" value="1.34"/>
          <Constant key="Parameter_4899" name="ka_cp_b2" value="0.334"/>
          <Constant key="Parameter_4900" name="ki_cp" value="0.21"/>
          <Constant key="Parameter_4904" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_61" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_389">
              <SourceParameter reference="Metabolite_84"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_381">
              <SourceParameter reference="ModelValue_245"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_368">
              <SourceParameter reference="Metabolite_96"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_385">
              <SourceParameter reference="ModelValue_136"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_373">
              <SourceParameter reference="ModelValue_202"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_387">
              <SourceParameter reference="ModelValue_203"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_377">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_67" name="CDH1A Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_67">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:22:42Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_85" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_98" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_70" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_73" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_95" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_96" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4905" name="CDH1T" value="0.808"/>
          <Constant key="Parameter_4908" name="e_h1_b2" value="2.35"/>
          <Constant key="Parameter_4907" name="e_h1_b5" value="9.73"/>
          <Constant key="Parameter_4906" name="e_h1_n2" value="1.56"/>
          <Constant key="Parameter_4909" name="e_h1_n3" value="3.75"/>
          <Constant key="Parameter_4910" name="gamma" value="2.22"/>
          <Constant key="Parameter_4911" name="ka_h1" value="0.241"/>
          <Constant key="Parameter_4912" name="ka_h1_14" value="32.2"/>
          <Constant key="Parameter_4913" name="ki_h1" value="0.144"/>
          <Constant key="Parameter_4914" name="ki_h1_e" value="0.215"/>
          <Constant key="Parameter_4915" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_62" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_394">
              <SourceParameter reference="Metabolite_98"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_416">
              <SourceParameter reference="Metabolite_85"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_386">
              <SourceParameter reference="ModelValue_246"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_414">
              <SourceParameter reference="Metabolite_96"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_410">
              <SourceParameter reference="Metabolite_95"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_406">
              <SourceParameter reference="Metabolite_73"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_402">
              <SourceParameter reference="Metabolite_70"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_412">
              <SourceParameter reference="ModelValue_211"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_408">
              <SourceParameter reference="ModelValue_210"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_404">
              <SourceParameter reference="ModelValue_209"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_400">
              <SourceParameter reference="ModelValue_208"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_390">
              <SourceParameter reference="ModelValue_134"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_383">
              <SourceParameter reference="ModelValue_204"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_392">
              <SourceParameter reference="ModelValue_205"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_396">
              <SourceParameter reference="ModelValue_206"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_398">
              <SourceParameter reference="ModelValue_207"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_375">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_68" name="NET1deP Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_68">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:26:49Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_86" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_98" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_87" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_96" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_100" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_89" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4916" name="NET1T" value="6.4"/>
          <Constant key="Parameter_4917" name="gamma" value="2.22"/>
          <Constant key="Parameter_4918" name="kdp_net" value="0.106"/>
          <Constant key="Parameter_4919" name="kdp_net_14" value="0.00663"/>
          <Constant key="Parameter_4920" name="kdp_net_px" value="83.3"/>
          <Constant key="Parameter_4921" name="kp_net" value="0.556"/>
          <Constant key="Parameter_4922" name="kp_net_15" value="0.00881"/>
          <Constant key="Parameter_4923" name="kp_net_b2" value="1.5"/>
          <Constant key="Parameter_4924" name="kp_net_en" value="6.88"/>
          <Constant key="Parameter_4925" name="signet" value="1.52"/>
        </ListOfConstants>
        <KineticLaw function="Function_63" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_397">
              <SourceParameter reference="Metabolite_98"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_429">
              <SourceParameter reference="Metabolite_89"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_421">
              <SourceParameter reference="Metabolite_96"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_425">
              <SourceParameter reference="Metabolite_100"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_413">
              <SourceParameter reference="ModelValue_247"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_431">
              <SourceParameter reference="Metabolite_86"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_379">
              <SourceParameter reference="Metabolite_87"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_417">
              <SourceParameter reference="ModelValue_134"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_405">
              <SourceParameter reference="ModelValue_212"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_401">
              <SourceParameter reference="ModelValue_213"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_393">
              <SourceParameter reference="ModelValue_214"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_388">
              <SourceParameter reference="ModelValue_215"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_427">
              <SourceParameter reference="ModelValue_218"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_419">
              <SourceParameter reference="ModelValue_216"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_423">
              <SourceParameter reference="ModelValue_217"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_409">
              <SourceParameter reference="ModelValue_139"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_69" name="PPX Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_69">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:29:21Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_87" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_99" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4926" name="PPXT" value="0.866"/>
          <Constant key="Parameter_4927" name="gamma" value="2.22"/>
          <Constant key="Parameter_4928" name="ka_px" value="0.055"/>
          <Constant key="Parameter_4929" name="ki_px" value="0.119"/>
          <Constant key="Parameter_4930" name="ki_px_p1" value="6.69"/>
          <Constant key="Parameter_4931" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_64" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_403">
              <SourceParameter reference="Metabolite_99"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_411">
              <SourceParameter reference="Metabolite_87"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_428">
              <SourceParameter reference="ModelValue_249"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_432">
              <SourceParameter reference="ModelValue_134"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_420">
              <SourceParameter reference="ModelValue_219"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_371">
              <SourceParameter reference="ModelValue_220"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_395">
              <SourceParameter reference="ModelValue_221"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_424">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_70" name="PDS1T Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_70">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:30:52Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_88" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfConstants>
          <Constant key="Parameter_4932" name="v" value="0.0467"/>
        </ListOfConstants>
        <KineticLaw function="Function_6" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_49">
              <SourceParameter reference="ModelValue_222"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_71" name="PDS1T Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_71">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:31:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_88" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_83" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_93" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4933" name="kd_pds" value="0.0144"/>
          <Constant key="Parameter_4934" name="kd_pds_20_i" value="0.125"/>
          <Constant key="Parameter_4935" name="ks_pds_20" value="3.04"/>
        </ListOfConstants>
        <KineticLaw function="Function_65" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_418">
              <SourceParameter reference="Metabolite_83"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_433">
              <SourceParameter reference="Metabolite_93"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_435">
              <SourceParameter reference="Metabolite_88"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_415">
              <SourceParameter reference="ModelValue_223"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_426">
              <SourceParameter reference="ModelValue_253"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_399">
              <SourceParameter reference="ModelValue_224"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_72" name="CDC15 Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_72">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:33:11Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_89" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_98" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_96" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4936" name="CDC15T" value="1.02"/>
          <Constant key="Parameter_4937" name="gamma" value="2.22"/>
          <Constant key="Parameter_4938" name="ka_15" value="0.709"/>
          <Constant key="Parameter_4939" name="ka_15_14" value="7.38"/>
          <Constant key="Parameter_4940" name="ki_15" value="0.894"/>
          <Constant key="Parameter_4941" name="ki_15_b2" value="2.16"/>
          <Constant key="Parameter_4812" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_66" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_441">
              <SourceParameter reference="Metabolite_98"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_449">
              <SourceParameter reference="Metabolite_89"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_430">
              <SourceParameter reference="ModelValue_251"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_447">
              <SourceParameter reference="Metabolite_96"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_436">
              <SourceParameter reference="ModelValue_134"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_437">
              <SourceParameter reference="ModelValue_225"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_439">
              <SourceParameter reference="ModelValue_226"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_443">
              <SourceParameter reference="ModelValue_227"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_445">
              <SourceParameter reference="ModelValue_228"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_391">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_73" name="TEM1 Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_73">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:34:57Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_90" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_92" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_99" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_87" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4813" name="TEM1T" value="1.29"/>
          <Constant key="Parameter_4814" name="gammatem" value="0.369"/>
          <Constant key="Parameter_4815" name="ka_tem" value="0.0848"/>
          <Constant key="Parameter_4816" name="ka_tem_lo" value="3.84"/>
          <Constant key="Parameter_4817" name="ka_tem_p1" value="0.0638"/>
          <Constant key="Parameter_4818" name="ki_tem" value="0.323"/>
          <Constant key="Parameter_4819" name="ki_tem_px" value="1.92"/>
          <Constant key="Parameter_4820" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_67" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_455">
              <SourceParameter reference="Metabolite_99"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_451">
              <SourceParameter reference="Metabolite_92"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_461">
              <SourceParameter reference="Metabolite_87"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_463">
              <SourceParameter reference="Metabolite_90"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_446">
              <SourceParameter reference="ModelValue_252"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_450">
              <SourceParameter reference="ModelValue_137"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_438">
              <SourceParameter reference="ModelValue_229"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_422">
              <SourceParameter reference="ModelValue_230"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_453">
              <SourceParameter reference="ModelValue_231"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_457">
              <SourceParameter reference="ModelValue_232"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_459">
              <SourceParameter reference="ModelValue_233"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_442">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_74" name="POLOT Synth" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_74">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:37:58Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_91" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_101" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4821" name="ks_lo" value="0.045"/>
          <Constant key="Parameter_4822" name="ks_lo_m1" value="0.0113"/>
        </ListOfConstants>
        <KineticLaw function="Function_68" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_456">
              <SourceParameter reference="Metabolite_101"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_464">
              <SourceParameter reference="ModelValue_234"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_460">
              <SourceParameter reference="ModelValue_235"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_75" name="POLOT Degr" reversible="false" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_75">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:38:40Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_91" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_85" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4823" name="kd_lo" value="0.00483"/>
          <Constant key="Parameter_4824" name="kd_lo_h1" value="0.139"/>
        </ListOfConstants>
        <KineticLaw function="Function_69" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_434">
              <SourceParameter reference="Metabolite_85"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_440">
              <SourceParameter reference="Metabolite_91"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_454">
              <SourceParameter reference="ModelValue_236"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_462">
              <SourceParameter reference="ModelValue_237"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_76" name="POLOA Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_76">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:39:37Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_92" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_91" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_96" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_85" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4942" name="gamma" value="2.22"/>
          <Constant key="Parameter_4943" name="ka_lo" value="0.0232"/>
          <Constant key="Parameter_7088" name="ka_lo_b2" value="1.11"/>
          <Constant key="Parameter_7087" name="kd_lo" value="0.00483"/>
          <Constant key="Parameter_7166" name="kd_lo_h1" value="0.139"/>
          <Constant key="Parameter_7165" name="ki_lo" value="0.965"/>
          <Constant key="Parameter_7164" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_70" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_480">
              <SourceParameter reference="Metabolite_85"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_470">
              <SourceParameter reference="Metabolite_96"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_474">
              <SourceParameter reference="Metabolite_92"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_452">
              <SourceParameter reference="Metabolite_91"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_444">
              <SourceParameter reference="ModelValue_134"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_466">
              <SourceParameter reference="ModelValue_238"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_468">
              <SourceParameter reference="ModelValue_239"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_476">
              <SourceParameter reference="ModelValue_236"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_478">
              <SourceParameter reference="ModelValue_237"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_472">
              <SourceParameter reference="ModelValue_240"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_448">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_77" name="CDC20A_APC Synth" reversible="true" fast="false">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Reaction_77">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-01-06T14:45:13Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <ListOfProducts>
          <Product metabolite="Metabolite_93" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_104" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_79" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_80" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_7163" name="gamma" value="2.22"/>
          <Constant key="Parameter_7170" name="ka_20" value="0.0104"/>
          <Constant key="Parameter_7169" name="kd_20" value="0.124"/>
          <Constant key="Parameter_7168" name="ki_20_ori" value="5.04"/>
          <Constant key="Parameter_7167" name="sig" value="9.63"/>
        </ListOfConstants>
        <KineticLaw function="Function_71" unitType="Default" scalingCompartment="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_485">
              <SourceParameter reference="Metabolite_93"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_477">
              <SourceParameter reference="Metabolite_104"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_407">
              <SourceParameter reference="Metabolite_79"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_483">
              <SourceParameter reference="Metabolite_80"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_481">
              <SourceParameter reference="ModelValue_134"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_469">
              <SourceParameter reference="ModelValue_198"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_487">
              <SourceParameter reference="ModelValue_197"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_465">
              <SourceParameter reference="ModelValue_201"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_473">
              <SourceParameter reference="ModelValue_138"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
    </ListOfReactions>
    <ListOfEvents>
      <Event key="Event_38" name="Cell division" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_38">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:31:04Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2],Reference=Concentration> lt 0.20000000000000001
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_68">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[V],Reference=Concentration>*&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[f],Reference=Value>
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_78">
            <Expression>
              0
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_105">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[DIV_COUNT],Reference=Concentration>+1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_106">
            <Expression>
              0
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_108">
            <Expression>
              0
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_107">
            <Expression>
              0
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_112">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[V],Reference=Concentration>
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_113">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[VDiv1],Reference=Concentration>
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_114">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[VDiv2],Reference=Concentration>
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_37" name="Origin relicensing" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_37">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:32:13Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2],Reference=Concentration>+&lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5],Reference=Concentration> lt 0.20000000000000001
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_79">
            <Expression>
              0
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_80">
            <Expression>
              0
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_36" name="Bud emergence" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_36">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:31:06Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[BUD],Reference=Concentration> ge 1
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_106">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_35" name="ORI activation" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="1">
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[ORI],Reference=Concentration> ge 1
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_107">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_34" name="SPN completion" delayAssignment="true" fireAtInitialTime="0" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_34">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:32:12Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SPN],Reference=Concentration> ge 1
        </TriggerExpression>
        <DelayExpression>
          0
        </DelayExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_108">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_33" name="CKI_max" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_33">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:32:48Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CKIT],Reference=Rate> lt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCKIT_sign],Reference=Concentration> eq 1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_110">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CKIT_peaks],Reference=Concentration>+1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_109">
            <Expression>
              -1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_32" name="CKI_min" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_32">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-09T15:55:32Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CKIT],Reference=Rate> gt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCKIT_sign],Reference=Concentration> eq -1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_109">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_31" name="Cln3_max" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_31">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:25:00Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLN3],Reference=Rate> lt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCln3_sign],Reference=Concentration> eq 1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_117">
            <Expression>
              -1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_126">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Cln3_peaks],Reference=Concentration>+1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_30" name="Cln3_min" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_30">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:26:53Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLN3],Reference=Rate> gt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCln3_sign],Reference=Concentration> eq -1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_117">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_29" name="Bck2_max" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_29">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:36:54Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[BCK2],Reference=Rate> lt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dBck2_sign],Reference=Concentration> eq 1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_118">
            <Expression>
              -1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_127">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Bck2_peaks],Reference=Concentration>+1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_28" name="Bck2_min" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_28">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:37:55Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[BCK2],Reference=Rate> gt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dBck2_sign],Reference=Concentration> eq -1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_118">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_27" name="Cln2_max" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_27">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:39:03Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLN2],Reference=Rate> lt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCln2_sign],Reference=Concentration> eq 1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_119">
            <Expression>
              -1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_128">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Cln2_peaks],Reference=Concentration>+1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_26" name="Cln2_min" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_26">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:40:24Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLN2],Reference=Rate> gt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCln2_sign],Reference=Concentration> eq -1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_119">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_25" name="Clb5T_max" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_25">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:42:18Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5T],Reference=Rate> lt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dClb5T_sign],Reference=Concentration> eq 1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_120">
            <Expression>
              -1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_129">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Clb5T_peaks],Reference=Concentration>+1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_39" name="Clb5T_min" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_39">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:48:30Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5T],Reference=Rate> gt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dClb5T_sign],Reference=Concentration> eq -1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_120">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_40" name="Clb2T_max" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_40">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:49:29Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2T],Reference=Rate> lt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dClb2T_sign],Reference=Concentration> eq 1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_121">
            <Expression>
              -1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_130">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Clb2T_peaks],Reference=Concentration>+1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_41" name="Clb2T_min" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_41">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:50:47Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2T],Reference=Rate> gt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dClb2T_sign],Reference=Concentration> eq -1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_121">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_42" name="SW15T_max" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_42">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:52:13Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SWI5T],Reference=Rate> lt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dSWI5T_sign],Reference=Concentration> eq 1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_122">
            <Expression>
              -1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_131">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SWI5T_peaks],Reference=Concentration>+1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_43" name="SWI5T_min" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_43">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:53:36Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SWI5T],Reference=Rate> gt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dSWI5T_sign],Reference=Concentration> eq -1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_122">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_44" name="CDC20T_max" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_44">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:55:40Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20T],Reference=Rate> lt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCDC20T_sign],Reference=Concentration> eq 1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_123">
            <Expression>
              -1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_132">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20T_peaks],Reference=Concentration>+1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_45" name="CDC20T_min" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_45">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:56:47Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20T],Reference=Rate> gt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCDC20T_sign],Reference=Concentration> eq -1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_123">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_46" name="PDS1T_max" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_46">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:58:11Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[PDS1T],Reference=Rate> lt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dPDS1T_sign],Reference=Concentration> eq 1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_124">
            <Expression>
              -1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_133">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[PDS1T_peaks],Reference=Concentration>+1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_47" name="PDS1T_min" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_47">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T14:59:34Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[PDS1T],Reference=Rate> gt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dPDS1T_sign],Reference=Concentration> eq -1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_124">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_48" name="POLOT_max" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_48">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T15:00:41Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[POLOT],Reference=Rate> lt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dPOLOT_sign],Reference=Concentration> eq 1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_125">
            <Expression>
              -1
            </Expression>
          </Assignment>
          <Assignment targetKey="Metabolite_134">
            <Expression>
              &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[POLOT_peaks],Reference=Concentration>+1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
      <Event key="Event_49" name="POLOT_min" fireAtInitialTime="1" persistentTrigger="1">
        <MiriamAnnotation>
<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Event_49">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-10T15:02:01Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
        <TriggerExpression>
          &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[POLOT],Reference=Rate> gt 0 and &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dPOLOT_sign],Reference=Concentration> eq -1
        </TriggerExpression>
        <ListOfAssignments>
          <Assignment targetKey="Metabolite_125">
            <Expression>
              1
            </Expression>
          </Assignment>
        </ListOfAssignments>
      </Event>
    </ListOfEvents>
    <ListOfModelParameterSets activeSet="ModelParameterSet_1">
      <ModelParameterSet key="ModelParameterSet_1" name="Initial State">
        <ModelParameterGroup cn="String=Initial Time" type="Group">
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle" value="0" type="Model" simulationType="time"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Compartment Sizes" type="Group">
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell]" value="1" type="Compartment" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Species Values" type="Group">
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[V]" value="2.7701847942200001e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[BCK2]" value="1.2586274391129999e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLN3]" value="4.3660521213250001e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[WHI5deP]" value="1.065918931689e+24" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SBFdeP]" value="4.0830115010460004e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLN2]" value="3.7698601764820001e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CKIT]" value="3.3483103164919997e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CKIP]" value="2.1077492999500002e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5T]" value="3.1254911047830002e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2T]" value="8.9127684683600002e+21" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[BUD]" value="1.4453138056800002e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[ORI]" value="3.9565465430489997e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SPN]" value="8.9729898769300001e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SWI5T]" value="7.3470118455400001e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20T]" value="4.8357791081710003e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20A_APCP]" value="1.56575662282e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[APCP]" value="8.9127684683599997e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDH1A]" value="1.6560887356750001e+24" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[NET1deP]" value="1.4874687916790001e+24" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[PPX]" value="1.9451514968110002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[PDS1T]" value="1.4212252422519999e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC15]" value="3.8903029936220005e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[TEM1]" value="4.293786431041e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[POLOT]" value="5.5825245744390006e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[POLOA]" value="6.0040744344289997e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20A_APC]" value="3.2218453584950002e+22" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[FuncSafety]" value="2.0721811535692416e+25" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB5]" value="5.1245127419632697e+21" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2]" value="1.4613254061860575e+21" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SBF]" value="0" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC14]" value="0" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[ESP1]" value="1.6861994399600015e+22" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[MEN]" value="4.293786431041e+22" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[MCM1]" value="1257024356.8376269" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SWI5A]" value="3.6723041258761773e+22" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20A_APCP_T]" value="4.8357791081710003e+22" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20A_APC_T]" value="0" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[DIV_COUNT]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[FLAG_BUD]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[FLAG_UDNA]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[FLAG_SPC]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCKIT_sign]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CKIT_peaks]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[OffsetTime]" value="-3.0110704285000002e+26" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[VDiv1]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[VDiv2]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[VDiv3]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Diff1]" value="0" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Diff2]" value="0" type="Species" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCln3_sign]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dBck2_sign]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCln2_sign]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dClb5T_sign]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dClb2T_sign]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dSWI5T_sign]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dCDC20T_sign]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dPDS1T_sign]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[dPOLOT_sign]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Cln3_peaks]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Bck2_peaks]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Cln2_peaks]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Clb5T_peaks]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[Clb2T_peaks]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[SWI5T_peaks]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CDC20T_peaks]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[PDS1T_peaks]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[POLOT_peaks]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Compartments[cell],Vector=Metabolites[CLB2CLB5]" value="6.5858381481493272e+21" type="Species" simulationType="assignment"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Global Quantities" type="Group">
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[mdt]" value="70" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[mu]" value="0.0099021025794277899" type="ModelValue" simulationType="assignment"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_n3]" value="1.1100000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[Dn3]" value="0.73199999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[Jn3]" value="4.2699999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_n3]" value="0.79400000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[gamma]" value="2.2200000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[gammaki]" value="12.9" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[gammacp]" value="1.3400000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[gammatem]" value="0.36899999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig]" value="9.6300000000000008" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[signet]" value="1.52" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_k2]" value="0.055300000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_k2]" value="3.0099999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_i5]" value="1.22" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_i5_14]" value="0.19500000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_i5]" value="0.0275" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_i5_n3]" value="6.0999999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_i5_k2]" value="23.699999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_i5_n2]" value="2.9700000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_i5_b5]" value="0.042200000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_bf]" value="2.9300000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_bf_b2]" value="9.3599999999999994" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_n2]" value="1e-08" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_n2_bf]" value="0.996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_n2]" value="0.032000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_ki]" value="0.0066299999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_ki_swi5]" value="0.088999999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_ki]" value="0.052400000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_kip]" value="0.89900000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_ki_e]" value="0.65000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ki_n3]" value="1.05" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ki_k2]" value="0.39700000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ki_n2]" value="19.5" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ki_b5]" value="2.3900000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ki_b2]" value="3.1200000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_ki]" value="0.83599999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_ki_14]" value="1.1100000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_b5]" value="0.00053799999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_b5_bf]" value="0.0178" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b5]" value="0.055599999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b5_20]" value="0.044499999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_b2]" value="0.00762" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_b2_m1]" value="0.031" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b2]" value="0.00298" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b2_20]" value="0.13600000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b2_h1]" value="0.66200000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_bud_e]" value="0.28699999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_bud_n3]" value="0.0077999999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_bud_n2]" value="1.1200000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_bud_b5]" value="3" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_bud_b2]" value="1.8899999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_bud]" value="0.058999999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_spn]" value="0.074300000000000005" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_spn]" value="0.038399999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[Jspn]" value="0.80900000000000005" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_ori_e]" value="1.8999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ori_b5]" value="5.04" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ori_b2]" value="0.124" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_ori]" value="0.081699999999999995" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_swi5]" value="0.0055799999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_swi5_m1]" value="0.038899999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_swi5]" value="0.042000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_swi5_14]" value="1.4099999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_swi5_b2]" value="0.028000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_m1_b2]" value="4.6500000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_m1]" value="3.3900000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_20]" value="0.022100000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_20_m1]" value="0.35399999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_20]" value="0.124" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_20]" value="0.0104" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b5_20_i]" value="0.0049800000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b2_20_i]" value="0.037400000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_20_ori]" value="5.04" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_cp_b2]" value="0.33400000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_cp]" value="0.20999999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_h1]" value="0.24099999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_h1_14]" value="32.200000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_h1]" value="0.14399999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_h1_e]" value="0.215" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_h1_n3]" value="3.75" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_h1_n2]" value="1.5600000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_h1_b5]" value="9.7300000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_h1_b2]" value="2.3500000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_net]" value="0.106" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_net_14]" value="0.0066299999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_net_px]" value="83.299999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_net]" value="0.55600000000000005" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_net_b2]" value="1.5" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_net_en]" value="6.8799999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_net_15]" value="0.0088100000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_px]" value="0.055" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_px]" value="0.11899999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_px_p1]" value="6.6900000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_pds]" value="0.046699999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_pds]" value="0.0144" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_pds_20]" value="3.04" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_15]" value="0.70899999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_15_14]" value="7.3799999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_15]" value="0.89400000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_15_b2]" value="2.1600000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_tem]" value="0.0848" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_tem_lo]" value="3.8399999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_tem_p1]" value="0.063799999999999996" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_tem]" value="0.32300000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_tem_px]" value="1.9199999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_lo]" value="0.044999999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_lo_m1]" value="0.011299999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_lo]" value="0.0048300000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_lo_h1]" value="0.13900000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_lo]" value="0.023199999999999998" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_lo_b2]" value="1.1100000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_lo]" value="0.96499999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kas_net]" value="5.6100000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[WHI5T]" value="2.1000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[SBFT]" value="0.46800000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[MCM1T]" value="0.28199999999999997" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[APCPT]" value="45.700000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[CDH1T]" value="0.80800000000000005" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[NET1T]" value="6.4000000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[CDC14T]" value="6.2300000000000004" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[PPXT]" value="0.86599999999999999" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[ESP1T]" value="0.26400000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[CDC15T]" value="1.02" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[TEM1T]" value="1.29" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_pds_20_i]" value="0.125" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[f]" value="0.40000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Values[phi_15]" value="500" type="ModelValue" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Kinetic Parameters" type="Group">
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Growth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Growth],ParameterGroup=Parameters,Parameter=mu" value="0.0099021025794277899" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[mu],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln3 Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln3 Synth],ParameterGroup=Parameters,Parameter=Dn3" value="0.73199999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[Dn3],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln3 Synth],ParameterGroup=Parameters,Parameter=Jn3" value="4.2699999999999996" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[Jn3],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln3 Synth],ParameterGroup=Parameters,Parameter=ks_n3" value="1.1100000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_n3],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln3 Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln3 Degr],ParameterGroup=Parameters,Parameter=k1" value="0.79400000000000004" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_n3],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Bck2 Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Bck2 Synth],ParameterGroup=Parameters,Parameter=ks_k2" value="0.055300000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_k2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Bck2 Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Bck2 Degr],ParameterGroup=Parameters,Parameter=k1" value="3.0099999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_k2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth],ParameterGroup=Parameters,Parameter=WHI5T" value="2.1000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[WHI5T],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth],ParameterGroup=Parameters,Parameter=gamma" value="2.2200000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth],ParameterGroup=Parameters,Parameter=kdp_i5" value="1.22" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_i5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth],ParameterGroup=Parameters,Parameter=kdp_i5_14" value="0.19500000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_i5_14],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth],ParameterGroup=Parameters,Parameter=kp_i5" value="0.0275" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_i5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth],ParameterGroup=Parameters,Parameter=kp_i5_b5" value="0.042200000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_i5_b5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth],ParameterGroup=Parameters,Parameter=kp_i5_k2" value="23.699999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_i5_k2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth],ParameterGroup=Parameters,Parameter=kp_i5_n2" value="2.9700000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_i5_n2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth],ParameterGroup=Parameters,Parameter=kp_i5_n3" value="6.0999999999999996" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_i5_n3],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[WHI5deP Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SBFdeP Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SBFdeP Synth],ParameterGroup=Parameters,Parameter=SBFT" value="0.46800000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[SBFT],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SBFdeP Synth],ParameterGroup=Parameters,Parameter=gamma" value="2.2200000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SBFdeP Synth],ParameterGroup=Parameters,Parameter=kdp_bf" value="2.9300000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_bf],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SBFdeP Synth],ParameterGroup=Parameters,Parameter=kp_bf_b2" value="9.3599999999999994" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_bf_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SBFdeP Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln2 Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln2 Synth],ParameterGroup=Parameters,Parameter=ks_n2" value="1e-08" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_n2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln2 Synth],ParameterGroup=Parameters,Parameter=ks_n2_bf" value="0.996" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_n2_bf],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln2 Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Cln2 Degr],ParameterGroup=Parameters,Parameter=k1" value="0.032000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_n2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIT Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIT Synth],ParameterGroup=Parameters,Parameter=ks_ki" value="0.0066299999999999996" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_ki],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIT Synth],ParameterGroup=Parameters,Parameter=ks_ki_swi5" value="0.088999999999999996" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_ki_swi5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIT Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIT Degr],ParameterGroup=Parameters,Parameter=kd_ki" value="0.052400000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_ki],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIT Degr],ParameterGroup=Parameters,Parameter=kd_kip" value="0.89900000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_kip],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=e_ki_b2" value="3.1200000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ki_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=e_ki_b5" value="2.3900000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ki_b5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=e_ki_k2" value="0.39700000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ki_k2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=e_ki_n2" value="19.5" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ki_n2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=e_ki_n3" value="1.05" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ki_n3],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=gammaki" value="12.9" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gammaki],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=kd_kip" value="0.89900000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_kip],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=kdp_ki" value="0.83599999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_ki],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=kdp_ki_14" value="1.1100000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_ki_14],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=kp_ki_e" value="0.65000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_ki_e],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CKIP Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb5T Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb5T Synth],ParameterGroup=Parameters,Parameter=ks_b5" value="0.00053799999999999996" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_b5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb5T Synth],ParameterGroup=Parameters,Parameter=ks_b5_bf" value="0.0178" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_b5_bf],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb5T Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb5T Degr],ParameterGroup=Parameters,Parameter=kd_b5" value="0.055599999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb5T Degr],ParameterGroup=Parameters,Parameter=kd_b5_20" value="0.044499999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b5_20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb5T Degr],ParameterGroup=Parameters,Parameter=kd_b5_20_i" value="0.0049800000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b5_20_i],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb2T Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb2T Synth],ParameterGroup=Parameters,Parameter=ks_b2" value="0.00762" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb2T Synth],ParameterGroup=Parameters,Parameter=ks_b2_m1" value="0.031" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_b2_m1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb2T Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb2T Degr],ParameterGroup=Parameters,Parameter=kd_b2" value="0.00298" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb2T Degr],ParameterGroup=Parameters,Parameter=kd_b2_20" value="0.13600000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b2_20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb2T Degr],ParameterGroup=Parameters,Parameter=kd_b2_20_i" value="0.037400000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b2_20_i],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[Clb2T Degr],ParameterGroup=Parameters,Parameter=kd_b2_h1" value="0.66200000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_b2_h1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[BUD Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[BUD Synth],ParameterGroup=Parameters,Parameter=e_bud_b2" value="1.8899999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_bud_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[BUD Synth],ParameterGroup=Parameters,Parameter=e_bud_b5" value="3" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_bud_b5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[BUD Synth],ParameterGroup=Parameters,Parameter=e_bud_n2" value="1.1200000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_bud_n2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[BUD Synth],ParameterGroup=Parameters,Parameter=e_bud_n3" value="0.0077999999999999996" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_bud_n3],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[BUD Synth],ParameterGroup=Parameters,Parameter=ks_bud_e" value="0.28699999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_bud_e],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[BUD Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[BUD Degr],ParameterGroup=Parameters,Parameter=k1" value="0.058999999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_bud],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[ORI Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[ORI Synth],ParameterGroup=Parameters,Parameter=e_ori_b2" value="0.124" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ori_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[ORI Synth],ParameterGroup=Parameters,Parameter=e_ori_b5" value="5.04" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_ori_b5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[ORI Synth],ParameterGroup=Parameters,Parameter=ks_ori_e" value="1.8999999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_ori_e],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[ORI Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[ORI Degr],ParameterGroup=Parameters,Parameter=k1" value="0.081699999999999995" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_ori],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SPN Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SPN Synth],ParameterGroup=Parameters,Parameter=Jspn" value="0.80900000000000005" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[Jspn],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SPN Synth],ParameterGroup=Parameters,Parameter=ks_spn" value="0.074300000000000005" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_spn],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SPN Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SPN Degr],ParameterGroup=Parameters,Parameter=k1" value="0.038399999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_spn],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SWI5T Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SWI5T Synth],ParameterGroup=Parameters,Parameter=ks_swi5" value="0.0055799999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_swi5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SWI5T Synth],ParameterGroup=Parameters,Parameter=ks_swi5_m1" value="0.038899999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_swi5_m1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SWI5T Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[SWI5T Degr],ParameterGroup=Parameters,Parameter=k1" value="0.042000000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_swi5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20T Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20T Synth],ParameterGroup=Parameters,Parameter=ks_20" value="0.022100000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20T Synth],ParameterGroup=Parameters,Parameter=ks_20_m1" value="0.35399999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_20_m1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20T Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20T Degr],ParameterGroup=Parameters,Parameter=k1" value="0.124" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APCP Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APCP Synth],ParameterGroup=Parameters,Parameter=gamma" value="2.2200000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APCP Synth],ParameterGroup=Parameters,Parameter=ka_20" value="0.0104" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APCP Synth],ParameterGroup=Parameters,Parameter=kd_20" value="0.124" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APCP Synth],ParameterGroup=Parameters,Parameter=ki_20_ori" value="5.04" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_20_ori],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APCP Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[APCP Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[APCP Synth],ParameterGroup=Parameters,Parameter=APCPT" value="45.700000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[APCPT],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[APCP Synth],ParameterGroup=Parameters,Parameter=gammacp" value="1.3400000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gammacp],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[APCP Synth],ParameterGroup=Parameters,Parameter=ka_cp_b2" value="0.33400000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_cp_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[APCP Synth],ParameterGroup=Parameters,Parameter=ki_cp" value="0.20999999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_cp],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[APCP Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=CDH1T" value="0.80800000000000005" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[CDH1T],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=e_h1_b2" value="2.3500000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_h1_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=e_h1_b5" value="9.7300000000000004" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_h1_b5],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=e_h1_n2" value="1.5600000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_h1_n2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=e_h1_n3" value="3.75" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[e_h1_n3],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=gamma" value="2.2200000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=ka_h1" value="0.24099999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_h1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=ka_h1_14" value="32.200000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_h1_14],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=ki_h1" value="0.14399999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_h1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=ki_h1_e" value="0.215" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_h1_e],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDH1A Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth],ParameterGroup=Parameters,Parameter=NET1T" value="6.4000000000000004" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[NET1T],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth],ParameterGroup=Parameters,Parameter=gamma" value="2.2200000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth],ParameterGroup=Parameters,Parameter=kdp_net" value="0.106" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_net],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth],ParameterGroup=Parameters,Parameter=kdp_net_14" value="0.0066299999999999996" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_net_14],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth],ParameterGroup=Parameters,Parameter=kdp_net_px" value="83.299999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kdp_net_px],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth],ParameterGroup=Parameters,Parameter=kp_net" value="0.55600000000000005" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_net],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth],ParameterGroup=Parameters,Parameter=kp_net_15" value="0.0088100000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_net_15],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth],ParameterGroup=Parameters,Parameter=kp_net_b2" value="1.5" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_net_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth],ParameterGroup=Parameters,Parameter=kp_net_en" value="6.8799999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kp_net_en],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[NET1deP Synth],ParameterGroup=Parameters,Parameter=signet" value="1.52" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[signet],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PPX Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PPX Synth],ParameterGroup=Parameters,Parameter=PPXT" value="0.86599999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[PPXT],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PPX Synth],ParameterGroup=Parameters,Parameter=gamma" value="2.2200000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PPX Synth],ParameterGroup=Parameters,Parameter=ka_px" value="0.055" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_px],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PPX Synth],ParameterGroup=Parameters,Parameter=ki_px" value="0.11899999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_px],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PPX Synth],ParameterGroup=Parameters,Parameter=ki_px_p1" value="6.6900000000000004" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_px_p1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PPX Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PDS1T Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PDS1T Synth],ParameterGroup=Parameters,Parameter=v" value="0.046699999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_pds],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PDS1T Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PDS1T Degr],ParameterGroup=Parameters,Parameter=kd_pds" value="0.0144" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_pds],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PDS1T Degr],ParameterGroup=Parameters,Parameter=kd_pds_20_i" value="0.125" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_pds_20_i],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[PDS1T Degr],ParameterGroup=Parameters,Parameter=ks_pds_20" value="3.04" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_pds_20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC15 Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC15 Synth],ParameterGroup=Parameters,Parameter=CDC15T" value="1.02" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[CDC15T],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC15 Synth],ParameterGroup=Parameters,Parameter=gamma" value="2.2200000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC15 Synth],ParameterGroup=Parameters,Parameter=ka_15" value="0.70899999999999996" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_15],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC15 Synth],ParameterGroup=Parameters,Parameter=ka_15_14" value="7.3799999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_15_14],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC15 Synth],ParameterGroup=Parameters,Parameter=ki_15" value="0.89400000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_15],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC15 Synth],ParameterGroup=Parameters,Parameter=ki_15_b2" value="2.1600000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_15_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC15 Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[TEM1 Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[TEM1 Synth],ParameterGroup=Parameters,Parameter=TEM1T" value="1.29" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[TEM1T],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[TEM1 Synth],ParameterGroup=Parameters,Parameter=gammatem" value="0.36899999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gammatem],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[TEM1 Synth],ParameterGroup=Parameters,Parameter=ka_tem" value="0.0848" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_tem],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[TEM1 Synth],ParameterGroup=Parameters,Parameter=ka_tem_lo" value="3.8399999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_tem_lo],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[TEM1 Synth],ParameterGroup=Parameters,Parameter=ka_tem_p1" value="0.063799999999999996" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_tem_p1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[TEM1 Synth],ParameterGroup=Parameters,Parameter=ki_tem" value="0.32300000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_tem],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[TEM1 Synth],ParameterGroup=Parameters,Parameter=ki_tem_px" value="1.9199999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_tem_px],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[TEM1 Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOT Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOT Synth],ParameterGroup=Parameters,Parameter=ks_lo" value="0.044999999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_lo],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOT Synth],ParameterGroup=Parameters,Parameter=ks_lo_m1" value="0.011299999999999999" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ks_lo_m1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOT Degr]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOT Degr],ParameterGroup=Parameters,Parameter=kd_lo" value="0.0048300000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_lo],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOT Degr],ParameterGroup=Parameters,Parameter=kd_lo_h1" value="0.13900000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_lo_h1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOA Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOA Synth],ParameterGroup=Parameters,Parameter=gamma" value="2.2200000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOA Synth],ParameterGroup=Parameters,Parameter=ka_lo" value="0.023199999999999998" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_lo],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOA Synth],ParameterGroup=Parameters,Parameter=ka_lo_b2" value="1.1100000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_lo_b2],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOA Synth],ParameterGroup=Parameters,Parameter=kd_lo" value="0.0048300000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_lo],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOA Synth],ParameterGroup=Parameters,Parameter=kd_lo_h1" value="0.13900000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_lo_h1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOA Synth],ParameterGroup=Parameters,Parameter=ki_lo" value="0.96499999999999997" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_lo],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[POLOA Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APC Synth]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APC Synth],ParameterGroup=Parameters,Parameter=gamma" value="2.2200000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[gamma],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APC Synth],ParameterGroup=Parameters,Parameter=ka_20" value="0.0104" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ka_20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APC Synth],ParameterGroup=Parameters,Parameter=kd_20" value="0.124" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[kd_20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APC Synth],ParameterGroup=Parameters,Parameter=ki_20_ori" value="5.04" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[ki_20_ori],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Yeast Cell Cycle,Vector=Reactions[CDC20A_APC Synth],ParameterGroup=Parameters,Parameter=sig" value="9.6300000000000008" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Yeast Cell Cycle,Vector=Values[sig],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
        </ModelParameterGroup>
      </ModelParameterSet>
    </ListOfModelParameterSets>
    <StateTemplate>
      <StateTemplateVariable objectReference="Model_1"/>
      <StateTemplateVariable objectReference="Metabolite_69"/>
      <StateTemplateVariable objectReference="Metabolite_70"/>
      <StateTemplateVariable objectReference="Metabolite_73"/>
      <StateTemplateVariable objectReference="Metabolite_74"/>
      <StateTemplateVariable objectReference="Metabolite_76"/>
      <StateTemplateVariable objectReference="Metabolite_77"/>
      <StateTemplateVariable objectReference="Metabolite_78"/>
      <StateTemplateVariable objectReference="Metabolite_79"/>
      <StateTemplateVariable objectReference="Metabolite_80"/>
      <StateTemplateVariable objectReference="Metabolite_81"/>
      <StateTemplateVariable objectReference="Metabolite_82"/>
      <StateTemplateVariable objectReference="Metabolite_88"/>
      <StateTemplateVariable objectReference="Metabolite_91"/>
      <StateTemplateVariable objectReference="Metabolite_68"/>
      <StateTemplateVariable objectReference="Metabolite_71"/>
      <StateTemplateVariable objectReference="Metabolite_83"/>
      <StateTemplateVariable objectReference="Metabolite_84"/>
      <StateTemplateVariable objectReference="Metabolite_85"/>
      <StateTemplateVariable objectReference="Metabolite_86"/>
      <StateTemplateVariable objectReference="Metabolite_87"/>
      <StateTemplateVariable objectReference="Metabolite_75"/>
      <StateTemplateVariable objectReference="Metabolite_89"/>
      <StateTemplateVariable objectReference="Metabolite_90"/>
      <StateTemplateVariable objectReference="Metabolite_72"/>
      <StateTemplateVariable objectReference="Metabolite_92"/>
      <StateTemplateVariable objectReference="Metabolite_93"/>
      <StateTemplateVariable objectReference="Metabolite_94"/>
      <StateTemplateVariable objectReference="Metabolite_95"/>
      <StateTemplateVariable objectReference="Metabolite_96"/>
      <StateTemplateVariable objectReference="Metabolite_97"/>
      <StateTemplateVariable objectReference="Metabolite_98"/>
      <StateTemplateVariable objectReference="Metabolite_99"/>
      <StateTemplateVariable objectReference="Metabolite_100"/>
      <StateTemplateVariable objectReference="Metabolite_101"/>
      <StateTemplateVariable objectReference="Metabolite_102"/>
      <StateTemplateVariable objectReference="Metabolite_103"/>
      <StateTemplateVariable objectReference="Metabolite_104"/>
      <StateTemplateVariable objectReference="Metabolite_111"/>
      <StateTemplateVariable objectReference="Metabolite_115"/>
      <StateTemplateVariable objectReference="Metabolite_116"/>
      <StateTemplateVariable objectReference="Metabolite_135"/>
      <StateTemplateVariable objectReference="ModelValue_129"/>
      <StateTemplateVariable objectReference="Metabolite_105"/>
      <StateTemplateVariable objectReference="Metabolite_106"/>
      <StateTemplateVariable objectReference="Metabolite_107"/>
      <StateTemplateVariable objectReference="Metabolite_108"/>
      <StateTemplateVariable objectReference="Metabolite_109"/>
      <StateTemplateVariable objectReference="Metabolite_110"/>
      <StateTemplateVariable objectReference="Metabolite_112"/>
      <StateTemplateVariable objectReference="Metabolite_113"/>
      <StateTemplateVariable objectReference="Metabolite_114"/>
      <StateTemplateVariable objectReference="Metabolite_117"/>
      <StateTemplateVariable objectReference="Metabolite_118"/>
      <StateTemplateVariable objectReference="Metabolite_119"/>
      <StateTemplateVariable objectReference="Metabolite_120"/>
      <StateTemplateVariable objectReference="Metabolite_121"/>
      <StateTemplateVariable objectReference="Metabolite_122"/>
      <StateTemplateVariable objectReference="Metabolite_123"/>
      <StateTemplateVariable objectReference="Metabolite_124"/>
      <StateTemplateVariable objectReference="Metabolite_125"/>
      <StateTemplateVariable objectReference="Metabolite_126"/>
      <StateTemplateVariable objectReference="Metabolite_127"/>
      <StateTemplateVariable objectReference="Metabolite_128"/>
      <StateTemplateVariable objectReference="Metabolite_129"/>
      <StateTemplateVariable objectReference="Metabolite_130"/>
      <StateTemplateVariable objectReference="Metabolite_131"/>
      <StateTemplateVariable objectReference="Metabolite_132"/>
      <StateTemplateVariable objectReference="Metabolite_133"/>
      <StateTemplateVariable objectReference="Metabolite_134"/>
      <StateTemplateVariable objectReference="Compartment_1"/>
      <StateTemplateVariable objectReference="ModelValue_128"/>
      <StateTemplateVariable objectReference="ModelValue_130"/>
      <StateTemplateVariable objectReference="ModelValue_131"/>
      <StateTemplateVariable objectReference="ModelValue_132"/>
      <StateTemplateVariable objectReference="ModelValue_133"/>
      <StateTemplateVariable objectReference="ModelValue_134"/>
      <StateTemplateVariable objectReference="ModelValue_135"/>
      <StateTemplateVariable objectReference="ModelValue_136"/>
      <StateTemplateVariable objectReference="ModelValue_137"/>
      <StateTemplateVariable objectReference="ModelValue_138"/>
      <StateTemplateVariable objectReference="ModelValue_139"/>
      <StateTemplateVariable objectReference="ModelValue_140"/>
      <StateTemplateVariable objectReference="ModelValue_141"/>
      <StateTemplateVariable objectReference="ModelValue_142"/>
      <StateTemplateVariable objectReference="ModelValue_143"/>
      <StateTemplateVariable objectReference="ModelValue_144"/>
      <StateTemplateVariable objectReference="ModelValue_145"/>
      <StateTemplateVariable objectReference="ModelValue_146"/>
      <StateTemplateVariable objectReference="ModelValue_147"/>
      <StateTemplateVariable objectReference="ModelValue_148"/>
      <StateTemplateVariable objectReference="ModelValue_149"/>
      <StateTemplateVariable objectReference="ModelValue_150"/>
      <StateTemplateVariable objectReference="ModelValue_151"/>
      <StateTemplateVariable objectReference="ModelValue_152"/>
      <StateTemplateVariable objectReference="ModelValue_153"/>
      <StateTemplateVariable objectReference="ModelValue_154"/>
      <StateTemplateVariable objectReference="ModelValue_155"/>
      <StateTemplateVariable objectReference="ModelValue_156"/>
      <StateTemplateVariable objectReference="ModelValue_157"/>
      <StateTemplateVariable objectReference="ModelValue_158"/>
      <StateTemplateVariable objectReference="ModelValue_159"/>
      <StateTemplateVariable objectReference="ModelValue_160"/>
      <StateTemplateVariable objectReference="ModelValue_161"/>
      <StateTemplateVariable objectReference="ModelValue_162"/>
      <StateTemplateVariable objectReference="ModelValue_163"/>
      <StateTemplateVariable objectReference="ModelValue_164"/>
      <StateTemplateVariable objectReference="ModelValue_165"/>
      <StateTemplateVariable objectReference="ModelValue_166"/>
      <StateTemplateVariable objectReference="ModelValue_167"/>
      <StateTemplateVariable objectReference="ModelValue_168"/>
      <StateTemplateVariable objectReference="ModelValue_169"/>
      <StateTemplateVariable objectReference="ModelValue_170"/>
      <StateTemplateVariable objectReference="ModelValue_171"/>
      <StateTemplateVariable objectReference="ModelValue_172"/>
      <StateTemplateVariable objectReference="ModelValue_173"/>
      <StateTemplateVariable objectReference="ModelValue_174"/>
      <StateTemplateVariable objectReference="ModelValue_175"/>
      <StateTemplateVariable objectReference="ModelValue_176"/>
      <StateTemplateVariable objectReference="ModelValue_177"/>
      <StateTemplateVariable objectReference="ModelValue_178"/>
      <StateTemplateVariable objectReference="ModelValue_179"/>
      <StateTemplateVariable objectReference="ModelValue_180"/>
      <StateTemplateVariable objectReference="ModelValue_181"/>
      <StateTemplateVariable objectReference="ModelValue_182"/>
      <StateTemplateVariable objectReference="ModelValue_183"/>
      <StateTemplateVariable objectReference="ModelValue_184"/>
      <StateTemplateVariable objectReference="ModelValue_185"/>
      <StateTemplateVariable objectReference="ModelValue_186"/>
      <StateTemplateVariable objectReference="ModelValue_187"/>
      <StateTemplateVariable objectReference="ModelValue_188"/>
      <StateTemplateVariable objectReference="ModelValue_189"/>
      <StateTemplateVariable objectReference="ModelValue_190"/>
      <StateTemplateVariable objectReference="ModelValue_191"/>
      <StateTemplateVariable objectReference="ModelValue_192"/>
      <StateTemplateVariable objectReference="ModelValue_193"/>
      <StateTemplateVariable objectReference="ModelValue_194"/>
      <StateTemplateVariable objectReference="ModelValue_195"/>
      <StateTemplateVariable objectReference="ModelValue_196"/>
      <StateTemplateVariable objectReference="ModelValue_197"/>
      <StateTemplateVariable objectReference="ModelValue_198"/>
      <StateTemplateVariable objectReference="ModelValue_199"/>
      <StateTemplateVariable objectReference="ModelValue_200"/>
      <StateTemplateVariable objectReference="ModelValue_201"/>
      <StateTemplateVariable objectReference="ModelValue_202"/>
      <StateTemplateVariable objectReference="ModelValue_203"/>
      <StateTemplateVariable objectReference="ModelValue_204"/>
      <StateTemplateVariable objectReference="ModelValue_205"/>
      <StateTemplateVariable objectReference="ModelValue_206"/>
      <StateTemplateVariable objectReference="ModelValue_207"/>
      <StateTemplateVariable objectReference="ModelValue_208"/>
      <StateTemplateVariable objectReference="ModelValue_209"/>
      <StateTemplateVariable objectReference="ModelValue_210"/>
      <StateTemplateVariable objectReference="ModelValue_211"/>
      <StateTemplateVariable objectReference="ModelValue_212"/>
      <StateTemplateVariable objectReference="ModelValue_213"/>
      <StateTemplateVariable objectReference="ModelValue_214"/>
      <StateTemplateVariable objectReference="ModelValue_215"/>
      <StateTemplateVariable objectReference="ModelValue_216"/>
      <StateTemplateVariable objectReference="ModelValue_217"/>
      <StateTemplateVariable objectReference="ModelValue_218"/>
      <StateTemplateVariable objectReference="ModelValue_219"/>
      <StateTemplateVariable objectReference="ModelValue_220"/>
      <StateTemplateVariable objectReference="ModelValue_221"/>
      <StateTemplateVariable objectReference="ModelValue_222"/>
      <StateTemplateVariable objectReference="ModelValue_223"/>
      <StateTemplateVariable objectReference="ModelValue_224"/>
      <StateTemplateVariable objectReference="ModelValue_225"/>
      <StateTemplateVariable objectReference="ModelValue_226"/>
      <StateTemplateVariable objectReference="ModelValue_227"/>
      <StateTemplateVariable objectReference="ModelValue_228"/>
      <StateTemplateVariable objectReference="ModelValue_229"/>
      <StateTemplateVariable objectReference="ModelValue_230"/>
      <StateTemplateVariable objectReference="ModelValue_231"/>
      <StateTemplateVariable objectReference="ModelValue_232"/>
      <StateTemplateVariable objectReference="ModelValue_233"/>
      <StateTemplateVariable objectReference="ModelValue_234"/>
      <StateTemplateVariable objectReference="ModelValue_235"/>
      <StateTemplateVariable objectReference="ModelValue_236"/>
      <StateTemplateVariable objectReference="ModelValue_237"/>
      <StateTemplateVariable objectReference="ModelValue_238"/>
      <StateTemplateVariable objectReference="ModelValue_239"/>
      <StateTemplateVariable objectReference="ModelValue_240"/>
      <StateTemplateVariable objectReference="ModelValue_241"/>
      <StateTemplateVariable objectReference="ModelValue_242"/>
      <StateTemplateVariable objectReference="ModelValue_243"/>
      <StateTemplateVariable objectReference="ModelValue_244"/>
      <StateTemplateVariable objectReference="ModelValue_245"/>
      <StateTemplateVariable objectReference="ModelValue_246"/>
      <StateTemplateVariable objectReference="ModelValue_247"/>
      <StateTemplateVariable objectReference="ModelValue_248"/>
      <StateTemplateVariable objectReference="ModelValue_249"/>
      <StateTemplateVariable objectReference="ModelValue_250"/>
      <StateTemplateVariable objectReference="ModelValue_251"/>
      <StateTemplateVariable objectReference="ModelValue_252"/>
      <StateTemplateVariable objectReference="ModelValue_253"/>
      <StateTemplateVariable objectReference="ModelValue_254"/>
      <StateTemplateVariable objectReference="ModelValue_255"/>
    </StateTemplate>
    <InitialState type="initialState">
      0 1.2586274391129999e+22 4.3660521213250001e+23 3.7698601764820001e+22 3.3483103164919997e+22 3.1254911047830002e+22 8.9127684683600002e+21 1.4453138056800002e+22 3.9565465430489997e+22 8.9729898769300001e+22 7.3470118455400001e+22 4.8357791081710003e+22 1.4212252422519999e+23 5.5825245744390006e+22 2.7701847942200001e+23 1.065918931689e+24 1.56575662282e+23 8.9127684683599997e+23 1.6560887356750001e+24 1.4874687916790001e+24 1.9451514968110002e+23 2.1077492999500002e+22 3.8903029936220005e+23 4.293786431041e+22 4.0830115010460004e+23 6.0040744344289997e+22 3.2218453584950002e+22 2.0721811535692416e+25 5.1245127419632697e+21 1.4613254061860575e+21 0 0 1.6861994399600015e+22 4.293786431041e+22 1257024356.8376269 3.6723041258761773e+22 4.8357791081710003e+22 0 -3.0110704285000002e+26 0 0 6.5858381481493272e+21 0.0099021025794277899 0 0 0 0 6.0221408570000002e+23 0 0 0 0 6.0221408570000002e+23 6.0221408570000002e+23 6.0221408570000002e+23 6.0221408570000002e+23 6.0221408570000002e+23 6.0221408570000002e+23 6.0221408570000002e+23 6.0221408570000002e+23 6.0221408570000002e+23 0 0 0 0 0 0 0 0 0 1 70 1.1100000000000001 0.73199999999999998 4.2699999999999996 0.79400000000000004 2.2200000000000002 12.9 1.3400000000000001 0.36899999999999999 9.6300000000000008 1.52 0.055300000000000002 3.0099999999999998 1.22 0.19500000000000001 0.0275 6.0999999999999996 23.699999999999999 2.9700000000000002 0.042200000000000001 2.9300000000000002 9.3599999999999994 1e-08 0.996 0.032000000000000001 0.0066299999999999996 0.088999999999999996 0.052400000000000002 0.89900000000000002 0.65000000000000002 1.05 0.39700000000000002 19.5 2.3900000000000001 3.1200000000000001 0.83599999999999997 1.1100000000000001 0.00053799999999999996 0.0178 0.055599999999999997 0.044499999999999998 0.00762 0.031 0.00298 0.13600000000000001 0.66200000000000003 0.28699999999999998 0.0077999999999999996 1.1200000000000001 3 1.8899999999999999 0.058999999999999997 0.074300000000000005 0.038399999999999997 0.80900000000000005 1.8999999999999999 5.04 0.124 0.081699999999999995 0.0055799999999999999 0.038899999999999997 0.042000000000000003 1.4099999999999999 0.028000000000000001 4.6500000000000004 3.3900000000000001 0.022100000000000002 0.35399999999999998 0.124 0.0104 0.0049800000000000001 0.037400000000000003 5.04 0.33400000000000002 0.20999999999999999 0.24099999999999999 32.200000000000003 0.14399999999999999 0.215 3.75 1.5600000000000001 9.7300000000000004 2.3500000000000001 0.106 0.0066299999999999996 83.299999999999997 0.55600000000000005 1.5 6.8799999999999999 0.0088100000000000001 0.055 0.11899999999999999 6.6900000000000004 0.046699999999999998 0.0144 3.04 0.70899999999999996 7.3799999999999999 0.89400000000000002 2.1600000000000001 0.0848 3.8399999999999999 0.063799999999999996 0.32300000000000001 1.9199999999999999 0.044999999999999998 0.011299999999999999 0.0048300000000000001 0.13900000000000001 0.023199999999999998 1.1100000000000001 0.96499999999999997 5.6100000000000003 2.1000000000000001 0.46800000000000003 0.28199999999999997 45.700000000000003 0.80800000000000005 6.4000000000000004 6.2300000000000004 0.86599999999999999 0.26400000000000001 1.02 1.29 0.125 0.40000000000000002 500 
    </InitialState>
  </Model>
  <ListOfTasks>
    <Task key="Task_26" name="Steady-State" type="steadyState" scheduled="false" updateModel="false">
      <Report reference="Report_17" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="JacobianRequested" type="bool" value="1"/>
        <Parameter name="StabilityAnalysisRequested" type="bool" value="1"/>
      </Problem>
      <Method name="Enhanced Newton" type="EnhancedNewton">
        <Parameter name="Resolution" type="unsignedFloat" value="1.0000000000000001e-09"/>
        <Parameter name="Derivation Factor" type="unsignedFloat" value="0.001"/>
        <Parameter name="Use Newton" type="bool" value="1"/>
        <Parameter name="Use Integration" type="bool" value="1"/>
        <Parameter name="Use Back Integration" type="bool" value="0"/>
        <Parameter name="Accept Negative Concentrations" type="bool" value="0"/>
        <Parameter name="Iteration Limit" type="unsignedInteger" value="50"/>
        <Parameter name="Maximum duration for forward integration" type="unsignedFloat" value="1000000000"/>
        <Parameter name="Maximum duration for backward integration" type="unsignedFloat" value="1000000"/>
      </Method>
    </Task>
    <Task key="Task_25" name="Time-Course" type="timeCourse" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="10000"/>
        <Parameter name="StepSize" type="float" value="0.10000000000000001"/>
        <Parameter name="Duration" type="float" value="1000"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
      </Problem>
      <Method name="Deterministic (LSODA)" type="Deterministic(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
    <Task key="Task_24" name="Scan" type="scan" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="Subtask" type="unsignedInteger" value="1"/>
        <ParameterGroup name="ScanItems">
        </ParameterGroup>
        <Parameter name="Output in subtask" type="bool" value="1"/>
        <Parameter name="Adjust initial conditions" type="bool" value="0"/>
      </Problem>
      <Method name="Scan Framework" type="ScanFramework">
      </Method>
    </Task>
    <Task key="Task_23" name="Elementary Flux Modes" type="fluxMode" scheduled="false" updateModel="false">
      <Report reference="Report_16" target="" append="1" confirmOverwrite="1"/>
      <Problem>
      </Problem>
      <Method name="EFM Algorithm" type="EFMAlgorithm">
      </Method>
    </Task>
    <Task key="Task_22" name="Optimization" type="optimization" scheduled="false" updateModel="false">
      <Report reference="Report_15" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Subtask" type="cn" value="CN=Root,Vector=TaskList[Steady-State]"/>
        <ParameterText name="ObjectiveExpression" type="expression">
          
        </ParameterText>
        <Parameter name="Maximize" type="bool" value="0"/>
        <Parameter name="Randomize Start Values" type="bool" value="0"/>
        <Parameter name="Calculate Statistics" type="bool" value="1"/>
        <ParameterGroup name="OptimizationItemList">
        </ParameterGroup>
        <ParameterGroup name="OptimizationConstraintList">
        </ParameterGroup>
      </Problem>
      <Method name="Random Search" type="RandomSearch">
        <Parameter name="Number of Iterations" type="unsignedInteger" value="100000"/>
        <Parameter name="Random Number Generator" type="unsignedInteger" value="1"/>
        <Parameter name="Seed" type="unsignedInteger" value="0"/>
      </Method>
    </Task>
    <Task key="Task_21" name="Parameter Estimation" type="parameterFitting" scheduled="false" updateModel="false">
      <Report reference="Report_14" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Maximize" type="bool" value="0"/>
        <Parameter name="Randomize Start Values" type="bool" value="0"/>
        <Parameter name="Calculate Statistics" type="bool" value="1"/>
        <ParameterGroup name="OptimizationItemList">
        </ParameterGroup>
        <ParameterGroup name="OptimizationConstraintList">
        </ParameterGroup>
        <Parameter name="Steady-State" type="cn" value="CN=Root,Vector=TaskList[Steady-State]"/>
        <Parameter name="Time-Course" type="cn" value="CN=Root,Vector=TaskList[Time-Course]"/>
        <Parameter name="Create Parameter Sets" type="bool" value="0"/>
        <ParameterGroup name="Experiment Set">
        </ParameterGroup>
        <ParameterGroup name="Validation Set">
          <Parameter name="Threshold" type="unsignedInteger" value="5"/>
          <Parameter name="Weight" type="unsignedFloat" value="1"/>
        </ParameterGroup>
      </Problem>
      <Method name="Evolutionary Programming" type="EvolutionaryProgram">
        <Parameter name="Number of Generations" type="unsignedInteger" value="200"/>
        <Parameter name="Population Size" type="unsignedInteger" value="20"/>
        <Parameter name="Random Number Generator" type="unsignedInteger" value="1"/>
        <Parameter name="Seed" type="unsignedInteger" value="0"/>
      </Method>
    </Task>
    <Task key="Task_20" name="Metabolic Control Analysis" type="metabolicControlAnalysis" scheduled="false" updateModel="false">
      <Report reference="Report_13" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Steady-State" type="key" value="Task_26"/>
      </Problem>
      <Method name="MCA Method (Reder)" type="MCAMethod(Reder)">
        <Parameter name="Modulation Factor" type="unsignedFloat" value="1.0000000000000001e-09"/>
        <Parameter name="Use Reder" type="bool" value="1"/>
        <Parameter name="Use Smallbone" type="bool" value="1"/>
      </Method>
    </Task>
    <Task key="Task_19" name="Lyapunov Exponents" type="lyapunovExponents" scheduled="false" updateModel="false">
      <Report reference="Report_12" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="ExponentNumber" type="unsignedInteger" value="3"/>
        <Parameter name="DivergenceRequested" type="bool" value="1"/>
        <Parameter name="TransientTime" type="float" value="0"/>
      </Problem>
      <Method name="Wolf Method" type="WolfMethod">
        <Parameter name="Orthonormalization Interval" type="unsignedFloat" value="1"/>
        <Parameter name="Overall time" type="unsignedFloat" value="1000"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
      </Method>
    </Task>
    <Task key="Task_18" name="Time Scale Separation Analysis" type="timeScaleSeparationAnalysis" scheduled="false" updateModel="false">
      <Report reference="Report_11" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
      </Problem>
      <Method name="ILDM (LSODA,Deuflhard)" type="TimeScaleSeparation(ILDM,Deuflhard)">
        <Parameter name="Deuflhard Tolerance" type="unsignedFloat" value="0.0001"/>
      </Method>
    </Task>
    <Task key="Task_17" name="Sensitivities" type="sensitivities" scheduled="false" updateModel="false">
      <Report reference="Report_10" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="SubtaskType" type="unsignedInteger" value="1"/>
        <ParameterGroup name="TargetFunctions">
          <Parameter name="SingleObject" type="cn" value=""/>
          <Parameter name="ObjectListType" type="unsignedInteger" value="7"/>
        </ParameterGroup>
        <ParameterGroup name="ListOfVariables">
          <ParameterGroup name="Variables">
            <Parameter name="SingleObject" type="cn" value=""/>
            <Parameter name="ObjectListType" type="unsignedInteger" value="41"/>
          </ParameterGroup>
          <ParameterGroup name="Variables">
            <Parameter name="SingleObject" type="cn" value=""/>
            <Parameter name="ObjectListType" type="unsignedInteger" value="0"/>
          </ParameterGroup>
        </ParameterGroup>
      </Problem>
      <Method name="Sensitivities Method" type="SensitivitiesMethod">
        <Parameter name="Delta factor" type="unsignedFloat" value="0.001"/>
        <Parameter name="Delta minimum" type="unsignedFloat" value="9.9999999999999998e-13"/>
      </Method>
    </Task>
    <Task key="Task_16" name="Moieties" type="moieties" scheduled="false" updateModel="false">
      <Problem>
      </Problem>
      <Method name="Householder Reduction" type="Householder">
      </Method>
    </Task>
    <Task key="Task_15" name="Cross Section" type="crosssection" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
        <Parameter name="LimitCrossings" type="bool" value="0"/>
        <Parameter name="NumCrossingsLimit" type="unsignedInteger" value="0"/>
        <Parameter name="LimitOutTime" type="bool" value="0"/>
        <Parameter name="LimitOutCrossings" type="bool" value="0"/>
        <Parameter name="PositiveDirection" type="bool" value="1"/>
        <Parameter name="NumOutCrossingsLimit" type="unsignedInteger" value="0"/>
        <Parameter name="LimitUntilConvergence" type="bool" value="0"/>
        <Parameter name="ConvergenceTolerance" type="float" value="9.9999999999999995e-07"/>
        <Parameter name="Threshold" type="float" value="0"/>
        <Parameter name="DelayOutputUntilConvergence" type="bool" value="0"/>
        <Parameter name="OutputConvergenceTolerance" type="float" value="9.9999999999999995e-07"/>
        <ParameterText name="TriggerExpression" type="expression">
          
        </ParameterText>
        <Parameter name="SingleVariable" type="cn" value=""/>
      </Problem>
      <Method name="Deterministic (LSODA)" type="Deterministic(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
    <Task key="Task_27" name="Linear Noise Approximation" type="linearNoiseApproximation" scheduled="false" updateModel="false">
      <Report reference="Report_9" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Steady-State" type="key" value="Task_26"/>
      </Problem>
      <Method name="Linear Noise Approximation" type="LinearNoiseApproximation">
      </Method>
    </Task>
  </ListOfTasks>
  <ListOfReports>
    <Report key="Report_17" name="Steady-State" taskType="steadyState" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Steady-State]"/>
      </Footer>
    </Report>
    <Report key="Report_16" name="Elementary Flux Modes" taskType="fluxMode" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Elementary Flux Modes],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_15" name="Optimization" taskType="optimization" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Object=Description"/>
        <Object cn="String=\[Function Evaluations\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Value\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Parameters\]"/>
      </Header>
      <Body>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Function Evaluations"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Best Value"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Best Parameters"/>
      </Body>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_14" name="Parameter Estimation" taskType="parameterFitting" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Object=Description"/>
        <Object cn="String=\[Function Evaluations\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Value\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Parameters\]"/>
      </Header>
      <Body>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Function Evaluations"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Best Value"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Best Parameters"/>
      </Body>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_13" name="Metabolic Control Analysis" taskType="metabolicControlAnalysis" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Metabolic Control Analysis],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Metabolic Control Analysis],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_12" name="Lyapunov Exponents" taskType="lyapunovExponents" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Lyapunov Exponents],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Lyapunov Exponents],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_11" name="Time Scale Separation Analysis" taskType="timeScaleSeparationAnalysis" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Time Scale Separation Analysis],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Time Scale Separation Analysis],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_10" name="Sensitivities" taskType="sensitivities" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Sensitivities],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Sensitivities],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_9" name="Linear Noise Approximation" taskType="linearNoiseApproximation" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Linear Noise Approximation],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Linear Noise Approximation],Object=Result"/>
      </Footer>
    </Report>
  </ListOfReports>
  <GUI>
  </GUI>
  <SBMLReference file="yeast_alpha.xml">
    <SBMLMap SBMLid="APCP" COPASIkey="Metabolite_84"/>
    <SBMLMap SBMLid="APCPT" COPASIkey="ModelValue_245"/>
    <SBMLMap SBMLid="APCP_Synth" COPASIkey="Reaction_66"/>
    <SBMLMap SBMLid="BCK2" COPASIkey="Metabolite_69"/>
    <SBMLMap SBMLid="BUD" COPASIkey="Metabolite_78"/>
    <SBMLMap SBMLid="BUD_Degr" COPASIkey="Reaction_56"/>
    <SBMLMap SBMLid="BUD_Synth" COPASIkey="Reaction_55"/>
    <SBMLMap SBMLid="Bck2_Degr" COPASIkey="Reaction_43"/>
    <SBMLMap SBMLid="Bck2_Synth" COPASIkey="Reaction_42"/>
    <SBMLMap SBMLid="Bck2_peaks" COPASIkey="Metabolite_127"/>
    <SBMLMap SBMLid="CDC14" COPASIkey="Metabolite_98"/>
    <SBMLMap SBMLid="CDC14T" COPASIkey="ModelValue_248"/>
    <SBMLMap SBMLid="CDC15" COPASIkey="Metabolite_89"/>
    <SBMLMap SBMLid="CDC15T" COPASIkey="ModelValue_251"/>
    <SBMLMap SBMLid="CDC15_Synth" COPASIkey="Reaction_72"/>
    <SBMLMap SBMLid="CDC20A" COPASIkey="Metabolite_83"/>
    <SBMLMap SBMLid="CDC20A_APC" COPASIkey="Metabolite_93"/>
    <SBMLMap SBMLid="CDC20A_APCP_T" COPASIkey="Metabolite_103"/>
    <SBMLMap SBMLid="CDC20A_APC_Synth" COPASIkey="Reaction_77"/>
    <SBMLMap SBMLid="CDC20A_APC_T" COPASIkey="Metabolite_104"/>
    <SBMLMap SBMLid="CDC20A_Synth" COPASIkey="Reaction_65"/>
    <SBMLMap SBMLid="CDC20T" COPASIkey="Metabolite_82"/>
    <SBMLMap SBMLid="CDC20T_Degr" COPASIkey="Reaction_64"/>
    <SBMLMap SBMLid="CDC20T_Synth" COPASIkey="Reaction_63"/>
    <SBMLMap SBMLid="CDC20T_peaks" COPASIkey="Metabolite_132"/>
    <SBMLMap SBMLid="CDH1A" COPASIkey="Metabolite_85"/>
    <SBMLMap SBMLid="CDH1A_Synth" COPASIkey="Reaction_67"/>
    <SBMLMap SBMLid="CDH1T" COPASIkey="ModelValue_246"/>
    <SBMLMap SBMLid="CKIP" COPASIkey="Metabolite_75"/>
    <SBMLMap SBMLid="CKIP_Synth" COPASIkey="Reaction_50"/>
    <SBMLMap SBMLid="CKIT" COPASIkey="Metabolite_74"/>
    <SBMLMap SBMLid="CKIT_Degr" COPASIkey="Reaction_49"/>
    <SBMLMap SBMLid="CKIT_Synth" COPASIkey="Reaction_48"/>
    <SBMLMap SBMLid="CKIT_peaks" COPASIkey="Metabolite_110"/>
    <SBMLMap SBMLid="CLB2" COPASIkey="Metabolite_96"/>
    <SBMLMap SBMLid="CLB2CLB5" COPASIkey="Metabolite_135"/>
    <SBMLMap SBMLid="CLB2T" COPASIkey="Metabolite_77"/>
    <SBMLMap SBMLid="CLB5" COPASIkey="Metabolite_95"/>
    <SBMLMap SBMLid="CLB5T" COPASIkey="Metabolite_76"/>
    <SBMLMap SBMLid="CLN2" COPASIkey="Metabolite_73"/>
    <SBMLMap SBMLid="CLN3" COPASIkey="Metabolite_70"/>
    <SBMLMap SBMLid="Clb2T_Degr" COPASIkey="Reaction_54"/>
    <SBMLMap SBMLid="Clb2T_Synth" COPASIkey="Reaction_53"/>
    <SBMLMap SBMLid="Clb2T_peaks" COPASIkey="Metabolite_130"/>
    <SBMLMap SBMLid="Clb5T_Degr" COPASIkey="Reaction_52"/>
    <SBMLMap SBMLid="Clb5T_Synth" COPASIkey="Reaction_51"/>
    <SBMLMap SBMLid="Clb5T_peaks" COPASIkey="Metabolite_129"/>
    <SBMLMap SBMLid="Cln2_Degr" COPASIkey="Reaction_47"/>
    <SBMLMap SBMLid="Cln2_Synth" COPASIkey="Reaction_46"/>
    <SBMLMap SBMLid="Cln2_peaks" COPASIkey="Metabolite_128"/>
    <SBMLMap SBMLid="Cln3_Degr" COPASIkey="Reaction_41"/>
    <SBMLMap SBMLid="Cln3_Synth" COPASIkey="Reaction_40"/>
    <SBMLMap SBMLid="Cln3_peaks" COPASIkey="Metabolite_126"/>
    <SBMLMap SBMLid="Constant_flux__irreversible" COPASIkey="Function_6"/>
    <SBMLMap SBMLid="DIV_COUNT" COPASIkey="Metabolite_105"/>
    <SBMLMap SBMLid="Diff1" COPASIkey="Metabolite_115"/>
    <SBMLMap SBMLid="Diff2" COPASIkey="Metabolite_116"/>
    <SBMLMap SBMLid="Dn3" COPASIkey="ModelValue_131"/>
    <SBMLMap SBMLid="ESP1" COPASIkey="Metabolite_99"/>
    <SBMLMap SBMLid="ESP1T" COPASIkey="ModelValue_250"/>
    <SBMLMap SBMLid="FLAG_BUD" COPASIkey="Metabolite_106"/>
    <SBMLMap SBMLid="FLAG_SPC" COPASIkey="Metabolite_108"/>
    <SBMLMap SBMLid="FLAG_UDNA" COPASIkey="Metabolite_107"/>
    <SBMLMap SBMLid="FuncSafety" COPASIkey="Metabolite_94"/>
    <SBMLMap SBMLid="Growth" COPASIkey="Reaction_39"/>
    <SBMLMap SBMLid="Heav" COPASIkey="Function_40"/>
    <SBMLMap SBMLid="Jn3" COPASIkey="ModelValue_132"/>
    <SBMLMap SBMLid="Jspn" COPASIkey="ModelValue_183"/>
    <SBMLMap SBMLid="MCM1" COPASIkey="Metabolite_101"/>
    <SBMLMap SBMLid="MCM1T" COPASIkey="ModelValue_244"/>
    <SBMLMap SBMLid="MEN" COPASIkey="Metabolite_100"/>
    <SBMLMap SBMLid="NET1T" COPASIkey="ModelValue_247"/>
    <SBMLMap SBMLid="NET1deP" COPASIkey="Metabolite_86"/>
    <SBMLMap SBMLid="NET1deP_Synth" COPASIkey="Reaction_68"/>
    <SBMLMap SBMLid="ORI" COPASIkey="Metabolite_79"/>
    <SBMLMap SBMLid="ORI_Degr" COPASIkey="Reaction_58"/>
    <SBMLMap SBMLid="ORI_Synth" COPASIkey="Reaction_57"/>
    <SBMLMap SBMLid="OffsetTime" COPASIkey="Metabolite_111"/>
    <SBMLMap SBMLid="PDS1T" COPASIkey="Metabolite_88"/>
    <SBMLMap SBMLid="PDS1T_Degr" COPASIkey="Reaction_71"/>
    <SBMLMap SBMLid="PDS1T_Synth" COPASIkey="Reaction_70"/>
    <SBMLMap SBMLid="PDS1T_peaks" COPASIkey="Metabolite_133"/>
    <SBMLMap SBMLid="POLOA" COPASIkey="Metabolite_92"/>
    <SBMLMap SBMLid="POLOA_Synth" COPASIkey="Reaction_76"/>
    <SBMLMap SBMLid="POLOT" COPASIkey="Metabolite_91"/>
    <SBMLMap SBMLid="POLOT_Degr" COPASIkey="Reaction_75"/>
    <SBMLMap SBMLid="POLOT_Synth" COPASIkey="Reaction_74"/>
    <SBMLMap SBMLid="POLOT_peaks" COPASIkey="Metabolite_134"/>
    <SBMLMap SBMLid="PPX" COPASIkey="Metabolite_87"/>
    <SBMLMap SBMLid="PPXT" COPASIkey="ModelValue_249"/>
    <SBMLMap SBMLid="PPX_Synth" COPASIkey="Reaction_69"/>
    <SBMLMap SBMLid="Rate_Law_for_APCP_Synth_1" COPASIkey="Function_61"/>
    <SBMLMap SBMLid="Rate_Law_for_BUD_Synth_1" COPASIkey="Function_55"/>
    <SBMLMap SBMLid="Rate_Law_for_Bck2_Synth_1" COPASIkey="Function_44"/>
    <SBMLMap SBMLid="Rate_Law_for_CDC15_Synth_1" COPASIkey="Function_66"/>
    <SBMLMap SBMLid="Rate_Law_for_CDC20A_APCP_Synth_1" COPASIkey="Function_60"/>
    <SBMLMap SBMLid="Rate_Law_for_CDC20A_APC_Synth_1" COPASIkey="Function_71"/>
    <SBMLMap SBMLid="Rate_Law_for_CDC20T_Synth_1" COPASIkey="Function_59"/>
    <SBMLMap SBMLid="Rate_Law_for_CDH1A_Synth_1" COPASIkey="Function_62"/>
    <SBMLMap SBMLid="Rate_Law_for_CKIP_Synth_1" COPASIkey="Function_50"/>
    <SBMLMap SBMLid="Rate_Law_for_CKIT_Degr_1" COPASIkey="Function_49"/>
    <SBMLMap SBMLid="Rate_Law_for_CKIT_Synth_1" COPASIkey="Function_48"/>
    <SBMLMap SBMLid="Rate_Law_for_Clb2T_Degr_1" COPASIkey="Function_54"/>
    <SBMLMap SBMLid="Rate_Law_for_Clb2T_Synth_1" COPASIkey="Function_53"/>
    <SBMLMap SBMLid="Rate_Law_for_Clb5T_Degr_1" COPASIkey="Function_52"/>
    <SBMLMap SBMLid="Rate_Law_for_Clb5T_Synth_1" COPASIkey="Function_51"/>
    <SBMLMap SBMLid="Rate_Law_for_Cln2_Synth_1" COPASIkey="Function_47"/>
    <SBMLMap SBMLid="Rate_Law_for_Cln3_Synth_1" COPASIkey="Function_43"/>
    <SBMLMap SBMLid="Rate_Law_for_Growth_1" COPASIkey="Function_42"/>
    <SBMLMap SBMLid="Rate_Law_for_NET1deP_Synth_1" COPASIkey="Function_63"/>
    <SBMLMap SBMLid="Rate_Law_for_ORI_Synth_1" COPASIkey="Function_56"/>
    <SBMLMap SBMLid="Rate_Law_for_PDS1T_Degr_1" COPASIkey="Function_65"/>
    <SBMLMap SBMLid="Rate_Law_for_POLOA_Synth_1" COPASIkey="Function_70"/>
    <SBMLMap SBMLid="Rate_Law_for_POLOT_Degr_1" COPASIkey="Function_69"/>
    <SBMLMap SBMLid="Rate_Law_for_POLOT_Synth_1" COPASIkey="Function_68"/>
    <SBMLMap SBMLid="Rate_Law_for_PPX_Synth_1" COPASIkey="Function_64"/>
    <SBMLMap SBMLid="Rate_Law_for_SBFdeP_Synth_1" COPASIkey="Function_46"/>
    <SBMLMap SBMLid="Rate_Law_for_SPN_Synth_1" COPASIkey="Function_57"/>
    <SBMLMap SBMLid="Rate_Law_for_SWI5T_Synth_1" COPASIkey="Function_58"/>
    <SBMLMap SBMLid="Rate_Law_for_TEM1_Synth_1" COPASIkey="Function_67"/>
    <SBMLMap SBMLid="Rate_Law_for_WHI5deP_Synth_1" COPASIkey="Function_45"/>
    <SBMLMap SBMLid="SBF" COPASIkey="Metabolite_97"/>
    <SBMLMap SBMLid="SBFT" COPASIkey="ModelValue_243"/>
    <SBMLMap SBMLid="SBFdeP" COPASIkey="Metabolite_72"/>
    <SBMLMap SBMLid="SBFdeP_Synth" COPASIkey="Reaction_45"/>
    <SBMLMap SBMLid="SPN" COPASIkey="Metabolite_80"/>
    <SBMLMap SBMLid="SPN_Degr" COPASIkey="Reaction_60"/>
    <SBMLMap SBMLid="SPN_Synth" COPASIkey="Reaction_59"/>
    <SBMLMap SBMLid="SWI5A" COPASIkey="Metabolite_102"/>
    <SBMLMap SBMLid="SWI5T" COPASIkey="Metabolite_81"/>
    <SBMLMap SBMLid="SWI5T_Degr" COPASIkey="Reaction_62"/>
    <SBMLMap SBMLid="SWI5T_Synth" COPASIkey="Reaction_61"/>
    <SBMLMap SBMLid="SWI5T_peaks" COPASIkey="Metabolite_131"/>
    <SBMLMap SBMLid="Sigmoid" COPASIkey="Function_41"/>
    <SBMLMap SBMLid="TEM1" COPASIkey="Metabolite_90"/>
    <SBMLMap SBMLid="TEM1T" COPASIkey="ModelValue_252"/>
    <SBMLMap SBMLid="TEM1_Synth" COPASIkey="Reaction_73"/>
    <SBMLMap SBMLid="V" COPASIkey="Metabolite_68"/>
    <SBMLMap SBMLid="VDiv1" COPASIkey="Metabolite_112"/>
    <SBMLMap SBMLid="VDiv2" COPASIkey="Metabolite_113"/>
    <SBMLMap SBMLid="VDiv3" COPASIkey="Metabolite_114"/>
    <SBMLMap SBMLid="WHI5T" COPASIkey="ModelValue_242"/>
    <SBMLMap SBMLid="WHI5deP" COPASIkey="Metabolite_71"/>
    <SBMLMap SBMLid="WHI5deP_Synth" COPASIkey="Reaction_44"/>
    <SBMLMap SBMLid="cell" COPASIkey="Compartment_1"/>
    <SBMLMap SBMLid="dBck2_sign" COPASIkey="Metabolite_118"/>
    <SBMLMap SBMLid="dCDC20T_sign" COPASIkey="Metabolite_123"/>
    <SBMLMap SBMLid="dCKIT_sign" COPASIkey="Metabolite_109"/>
    <SBMLMap SBMLid="dClb2T_sign" COPASIkey="Metabolite_121"/>
    <SBMLMap SBMLid="dClb5T_sign" COPASIkey="Metabolite_120"/>
    <SBMLMap SBMLid="dCln2_sign" COPASIkey="Metabolite_119"/>
    <SBMLMap SBMLid="dCln3_sign" COPASIkey="Metabolite_117"/>
    <SBMLMap SBMLid="dPDS1T_sign" COPASIkey="Metabolite_124"/>
    <SBMLMap SBMLid="dPOLOT_sign" COPASIkey="Metabolite_125"/>
    <SBMLMap SBMLid="dSWI5T_sign" COPASIkey="Metabolite_122"/>
    <SBMLMap SBMLid="e_bud_b2" COPASIkey="ModelValue_179"/>
    <SBMLMap SBMLid="e_bud_b5" COPASIkey="ModelValue_178"/>
    <SBMLMap SBMLid="e_bud_n2" COPASIkey="ModelValue_177"/>
    <SBMLMap SBMLid="e_bud_n3" COPASIkey="ModelValue_176"/>
    <SBMLMap SBMLid="e_h1_b2" COPASIkey="ModelValue_211"/>
    <SBMLMap SBMLid="e_h1_b5" COPASIkey="ModelValue_210"/>
    <SBMLMap SBMLid="e_h1_n2" COPASIkey="ModelValue_209"/>
    <SBMLMap SBMLid="e_h1_n3" COPASIkey="ModelValue_208"/>
    <SBMLMap SBMLid="e_ki_b2" COPASIkey="ModelValue_163"/>
    <SBMLMap SBMLid="e_ki_b5" COPASIkey="ModelValue_162"/>
    <SBMLMap SBMLid="e_ki_k2" COPASIkey="ModelValue_160"/>
    <SBMLMap SBMLid="e_ki_n2" COPASIkey="ModelValue_161"/>
    <SBMLMap SBMLid="e_ki_n3" COPASIkey="ModelValue_159"/>
    <SBMLMap SBMLid="e_ori_b2" COPASIkey="ModelValue_186"/>
    <SBMLMap SBMLid="e_ori_b5" COPASIkey="ModelValue_185"/>
    <SBMLMap SBMLid="f" COPASIkey="ModelValue_254"/>
    <SBMLMap SBMLid="gamma" COPASIkey="ModelValue_134"/>
    <SBMLMap SBMLid="gammacp" COPASIkey="ModelValue_136"/>
    <SBMLMap SBMLid="gammaki" COPASIkey="ModelValue_135"/>
    <SBMLMap SBMLid="gammatem" COPASIkey="ModelValue_137"/>
    <SBMLMap SBMLid="ka_15" COPASIkey="ModelValue_225"/>
    <SBMLMap SBMLid="ka_15_14" COPASIkey="ModelValue_226"/>
    <SBMLMap SBMLid="ka_20" COPASIkey="ModelValue_198"/>
    <SBMLMap SBMLid="ka_cp_b2" COPASIkey="ModelValue_202"/>
    <SBMLMap SBMLid="ka_h1" COPASIkey="ModelValue_204"/>
    <SBMLMap SBMLid="ka_h1_14" COPASIkey="ModelValue_205"/>
    <SBMLMap SBMLid="ka_lo" COPASIkey="ModelValue_238"/>
    <SBMLMap SBMLid="ka_lo_b2" COPASIkey="ModelValue_239"/>
    <SBMLMap SBMLid="ka_m1_b2" COPASIkey="ModelValue_193"/>
    <SBMLMap SBMLid="ka_px" COPASIkey="ModelValue_219"/>
    <SBMLMap SBMLid="ka_swi5_14" COPASIkey="ModelValue_191"/>
    <SBMLMap SBMLid="ka_tem" COPASIkey="ModelValue_229"/>
    <SBMLMap SBMLid="ka_tem_lo" COPASIkey="ModelValue_230"/>
    <SBMLMap SBMLid="ka_tem_p1" COPASIkey="ModelValue_231"/>
    <SBMLMap SBMLid="kas_net" COPASIkey="ModelValue_241"/>
    <SBMLMap SBMLid="kd_20" COPASIkey="ModelValue_197"/>
    <SBMLMap SBMLid="kd_b2" COPASIkey="ModelValue_172"/>
    <SBMLMap SBMLid="kd_b2_20" COPASIkey="ModelValue_173"/>
    <SBMLMap SBMLid="kd_b2_20_i" COPASIkey="ModelValue_200"/>
    <SBMLMap SBMLid="kd_b2_h1" COPASIkey="ModelValue_174"/>
    <SBMLMap SBMLid="kd_b5" COPASIkey="ModelValue_168"/>
    <SBMLMap SBMLid="kd_b5_20" COPASIkey="ModelValue_169"/>
    <SBMLMap SBMLid="kd_b5_20_i" COPASIkey="ModelValue_199"/>
    <SBMLMap SBMLid="kd_bud" COPASIkey="ModelValue_180"/>
    <SBMLMap SBMLid="kd_k2" COPASIkey="ModelValue_141"/>
    <SBMLMap SBMLid="kd_ki" COPASIkey="ModelValue_156"/>
    <SBMLMap SBMLid="kd_kip" COPASIkey="ModelValue_157"/>
    <SBMLMap SBMLid="kd_lo" COPASIkey="ModelValue_236"/>
    <SBMLMap SBMLid="kd_lo_h1" COPASIkey="ModelValue_237"/>
    <SBMLMap SBMLid="kd_n2" COPASIkey="ModelValue_153"/>
    <SBMLMap SBMLid="kd_n3" COPASIkey="ModelValue_133"/>
    <SBMLMap SBMLid="kd_ori" COPASIkey="ModelValue_187"/>
    <SBMLMap SBMLid="kd_pds" COPASIkey="ModelValue_223"/>
    <SBMLMap SBMLid="kd_pds_20_i" COPASIkey="ModelValue_253"/>
    <SBMLMap SBMLid="kd_spn" COPASIkey="ModelValue_182"/>
    <SBMLMap SBMLid="kd_swi5" COPASIkey="ModelValue_190"/>
    <SBMLMap SBMLid="kdp_bf" COPASIkey="ModelValue_149"/>
    <SBMLMap SBMLid="kdp_i5" COPASIkey="ModelValue_142"/>
    <SBMLMap SBMLid="kdp_i5_14" COPASIkey="ModelValue_143"/>
    <SBMLMap SBMLid="kdp_ki" COPASIkey="ModelValue_164"/>
    <SBMLMap SBMLid="kdp_ki_14" COPASIkey="ModelValue_165"/>
    <SBMLMap SBMLid="kdp_net" COPASIkey="ModelValue_212"/>
    <SBMLMap SBMLid="kdp_net_14" COPASIkey="ModelValue_213"/>
    <SBMLMap SBMLid="kdp_net_px" COPASIkey="ModelValue_214"/>
    <SBMLMap SBMLid="ki_15" COPASIkey="ModelValue_227"/>
    <SBMLMap SBMLid="ki_15_b2" COPASIkey="ModelValue_228"/>
    <SBMLMap SBMLid="ki_20_ori" COPASIkey="ModelValue_201"/>
    <SBMLMap SBMLid="ki_cp" COPASIkey="ModelValue_203"/>
    <SBMLMap SBMLid="ki_h1" COPASIkey="ModelValue_206"/>
    <SBMLMap SBMLid="ki_h1_e" COPASIkey="ModelValue_207"/>
    <SBMLMap SBMLid="ki_lo" COPASIkey="ModelValue_240"/>
    <SBMLMap SBMLid="ki_m1" COPASIkey="ModelValue_194"/>
    <SBMLMap SBMLid="ki_px" COPASIkey="ModelValue_220"/>
    <SBMLMap SBMLid="ki_px_p1" COPASIkey="ModelValue_221"/>
    <SBMLMap SBMLid="ki_swi5_b2" COPASIkey="ModelValue_192"/>
    <SBMLMap SBMLid="ki_tem" COPASIkey="ModelValue_232"/>
    <SBMLMap SBMLid="ki_tem_px" COPASIkey="ModelValue_233"/>
    <SBMLMap SBMLid="kp_bf_b2" COPASIkey="ModelValue_150"/>
    <SBMLMap SBMLid="kp_i5" COPASIkey="ModelValue_144"/>
    <SBMLMap SBMLid="kp_i5_b5" COPASIkey="ModelValue_148"/>
    <SBMLMap SBMLid="kp_i5_k2" COPASIkey="ModelValue_146"/>
    <SBMLMap SBMLid="kp_i5_n2" COPASIkey="ModelValue_147"/>
    <SBMLMap SBMLid="kp_i5_n3" COPASIkey="ModelValue_145"/>
    <SBMLMap SBMLid="kp_ki_e" COPASIkey="ModelValue_158"/>
    <SBMLMap SBMLid="kp_net" COPASIkey="ModelValue_215"/>
    <SBMLMap SBMLid="kp_net_15" COPASIkey="ModelValue_218"/>
    <SBMLMap SBMLid="kp_net_b2" COPASIkey="ModelValue_216"/>
    <SBMLMap SBMLid="kp_net_en" COPASIkey="ModelValue_217"/>
    <SBMLMap SBMLid="ks_20" COPASIkey="ModelValue_195"/>
    <SBMLMap SBMLid="ks_20_m1" COPASIkey="ModelValue_196"/>
    <SBMLMap SBMLid="ks_b2" COPASIkey="ModelValue_170"/>
    <SBMLMap SBMLid="ks_b2_m1" COPASIkey="ModelValue_171"/>
    <SBMLMap SBMLid="ks_b5" COPASIkey="ModelValue_166"/>
    <SBMLMap SBMLid="ks_b5_bf" COPASIkey="ModelValue_167"/>
    <SBMLMap SBMLid="ks_bud_e" COPASIkey="ModelValue_175"/>
    <SBMLMap SBMLid="ks_k2" COPASIkey="ModelValue_140"/>
    <SBMLMap SBMLid="ks_ki" COPASIkey="ModelValue_154"/>
    <SBMLMap SBMLid="ks_ki_swi5" COPASIkey="ModelValue_155"/>
    <SBMLMap SBMLid="ks_lo" COPASIkey="ModelValue_234"/>
    <SBMLMap SBMLid="ks_lo_m1" COPASIkey="ModelValue_235"/>
    <SBMLMap SBMLid="ks_n2" COPASIkey="ModelValue_151"/>
    <SBMLMap SBMLid="ks_n2_bf" COPASIkey="ModelValue_152"/>
    <SBMLMap SBMLid="ks_n3" COPASIkey="ModelValue_130"/>
    <SBMLMap SBMLid="ks_ori_e" COPASIkey="ModelValue_184"/>
    <SBMLMap SBMLid="ks_pds" COPASIkey="ModelValue_222"/>
    <SBMLMap SBMLid="ks_pds_20" COPASIkey="ModelValue_224"/>
    <SBMLMap SBMLid="ks_spn" COPASIkey="ModelValue_181"/>
    <SBMLMap SBMLid="ks_swi5" COPASIkey="ModelValue_188"/>
    <SBMLMap SBMLid="ks_swi5_m1" COPASIkey="ModelValue_189"/>
    <SBMLMap SBMLid="mdt" COPASIkey="ModelValue_128"/>
    <SBMLMap SBMLid="mu" COPASIkey="ModelValue_129"/>
    <SBMLMap SBMLid="sig" COPASIkey="ModelValue_138"/>
    <SBMLMap SBMLid="signet" COPASIkey="ModelValue_139"/>
  </SBMLReference>
  <ListOfUnitDefinitions>
    <UnitDefinition key="Unit_0" name="meter" symbol="m">
      <Expression>
        m
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_2" name="second" symbol="s">
      <Expression>
        s
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_6" name="Avogadro" symbol="Avogadro">
      <Expression>
        Avogadro
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_8" name="item" symbol="#">
      <Expression>
        #
      </Expression>
    </UnitDefinition>
  </ListOfUnitDefinitions>
</COPASI>
