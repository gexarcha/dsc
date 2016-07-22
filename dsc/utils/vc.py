from git import Repo
import os

def is_dirty():
	# print __file__
	repo_path = os.path.abspath(__file__)[:-17] 
	repo = Repo(repo_path)
	return repo.is_dirty()

def tag(run_tag="run_"):
	repo_path = os.path.abspath(__file__)[:-17] 
	repo = Repo(repo_path)
	ts=[]
	tag_id=1
	if run_tag=="run_":
		for t in repo.tags:
			if t.find("run_")==0:
				ts.append(int(t[4:]))
		tag_id = sorted(ts)[-1]+1
		run_tag = run_tag+str(tag_id)
	print "tagging run with: {}".format(run_tag)
	repo.create_tag(run_tag,message="tag created automatically")


def commit(msg='automated commit by pulp/utils/vc.py:commit'):
	repo_path = os.path.abspath(__file__)[:-17] 
	repo = Repo(repo_path)
	repo.index.commit(msg)

def add_all():
	repo_path = os.path.abspath(__file__)[:-17] 
	repo = Repo(repo_path)
	repo.index.add([repo.index.diff(None)[n].a_path for n in range(len(repo.index.diff(None)))])

def autocommit(comm,commit_msg='automated commit by pulp/utils/vc.py:autocommit',run_tag='run_'):
	if comm.rank==0:
		if is_dirty():
			add_all()
			commit(commit_msg)
			tag(run_tag)
			
class VClog(object):
	repo_path = os.path.abspath(__file__)[:-17] 
	repo = Repo(repo_path)
	def __init__(self):
		pass

	def autocommit(self,comm,commit_msg='automated commit by pulp/utils/vc.py:autocommit',run_tag='run_'):
		if comm.rank==0:
			if self.repo.is_dirty():
				self.add_all()
				self.commit(commit_msg+' '+run_tag)
				self.repo.index.update()
				self.tag(run_tag)

	def tag(self, run_tag="run_"):
		ts=[]
		tag_id=1
		if run_tag=="run_":
			for t in self.repo.tags:
				if t.find("run_")==0:
					ts.append(int(t[4:]))
			tag_id = sorted(ts)[-1]+1
			run_tag = run_tag+str(tag_id)
		print "tagging run with: {}".format(run_tag)
		self.repo.create_tag(run_tag,message="tag created automatically")


	def commit(self, msg='automated commit by pulp/utils/vc.py:commit'):
		self.repo.index.commit(msg)

	def add_all(self):
		self.repo.index.add([self.repo.index.diff(None)[n].a_path for n in range(len(self.repo.index.diff(None)))])