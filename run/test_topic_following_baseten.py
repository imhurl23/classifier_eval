from run import topic_following_baseten as lifecycle


def test_second_activation_failure_still_shuts_down_both_deployments(tmp_path,monkeypatch):
    def load():
        monkeypatch.setenv('BASETEN_API_KEY','fake-test-key')
        for arm,prefix in [('tev1_native','BASETEN_TEV1'),('qwen35_4b_letters','BASETEN_QWEN35')]:
            model,deployment=lifecycle.ALLOWLIST[arm]
            monkeypatch.setenv(prefix+'_MODEL_ID',model);monkeypatch.setenv(prefix+'_DEPLOYMENT_ID',deployment)
    monkeypatch.setattr(lifecycle,'load_local_service_keys',load)
    calls=[]
    class Response:
        def __init__(self,payload,error=False):self.payload=payload;self.error=error
        def raise_for_status(self):
            if self.error:raise RuntimeError('activation failed')
        def json(self):return self.payload
    def request(method,url,**kwargs):
        calls.append((method,url))
        if url.endswith('/activate'):
            return Response({},error=lifecycle.ALLOWLIST['qwen35_4b_letters'][1] in url)
        if url.endswith('/deactivate'):return Response({})
        model,deployment=url.split('/models/')[1].split('/deployments/')
        return Response({'model_id':model,'id':deployment,'status':'INACTIVE','active_replica_count':0,
                         'instance_type_name':'L4:2x24x96 - 2 L4 GPU, 48 GiB video memory, 24 vCPU, 96 GiB',
                         'autoscaling_settings':{'min_replica':0,'max_replica':1}})
    monkeypatch.setattr(lifecycle.requests,'request',request)
    result=lifecycle.run(tmp_path)
    assert result['status']=='stopped'
    assert len([url for method,url in calls if url.endswith('/deactivate')])==2
    assert all(d['final_status']=='INACTIVE' and d['final_replicas']==0 for d in result['deployments'].values())
