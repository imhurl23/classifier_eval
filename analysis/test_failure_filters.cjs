// Offline tests of the actual pure filter predicates. No browser, DOM engine,
// page navigation, network access, or source-example code execution is used.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'output/failures-2026-10-12/failure-explorer.html'),'utf8');
const payload=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const D=JSON.parse(payload);assert.equal(D.cases.length,10294);assert.equal(D.cases.reduce((n,c)=>n+c.predictions.length,0),128899);
assert(!payload.includes('</script>'));assert(D.urls.filter(Boolean).every(u=>u.startsWith('https://www.braintrust.dev/app/')));
const template=fs.readFileSync(path.join(__dirname,'failure_explorer.html'),'utf8');
const begin=template.indexOf('function compared(c)'),end=template.indexOf('function update()');
const controls={};for(const id of ['source','unit','target','model','mode','tag','adversarial','length','contrast','agreement','search','errors','high','references'])controls[id]={value:'',checked:false};
const sandbox={D,$:id=>controls[id]};vm.createContext(sandbox);vm.runInContext(template.slice(begin,end),sandbox);
controls.source.value='wildguardtest';controls.unit.value='conversation';controls.target.value='response_refusal';controls.model.value='qwen35_4b_json';controls.errors.checked=true;
assert.equal(D.cases.filter(sandbox.qualifies).length,434);
controls.mode.value='false_refusal';assert.equal(D.cases.filter(sandbox.qualifies).length,407);
controls.mode.value='missed_refusal';assert.equal(D.cases.filter(sandbox.qualifies).length,27);
controls.mode.value='';controls.high.checked=true;assert.equal(D.cases.filter(sandbox.qualifies).length,0); // label-only JSON has no confidence
controls.high.checked=false;controls.model.value='qwen3guard_8b';controls.target.value='interaction_harm';
assert.equal(D.cases.filter(sandbox.qualifies).length,0);controls.references.checked=true;assert(D.cases.filter(sandbox.qualifies).length>0);
controls.references.checked=false;controls.model.value='';controls.mode.value='classifier_refusal';controls.high.checked=true;
assert.equal(D.cases.filter(sandbox.qualifies).length,0); // high confidence must belong to SAME failing prediction
console.log('Offline filter checks passed: scope, 434 refusal errors, 407 false refusals, 27 misses, confidence missingness, and reference isolation.');
